"""One molecular wavefunction assembly with FermiNet or Psiformer interactions."""

import torch
from torch import nn

from .determinants import signed_determinant_sum
from .envelope import NuclearEnvelope
from .features import ElectronFeatureEmbedding
from .geometry import GeometricContext
from .jastrow import KatoCuspJastrow
from .layers import FermiInteractionLayer
from .physical_config import (
    cusp_defaults,
    resolve_spin,
    pair_spin_for,
    validate_pair_cusp,
)
from .psiformer_attention import PsiformerAttentionBlock


class FermiNet(nn.Module):
    """Same-spin antisymmetric molecular wavefunction.

    ``forward_log_psi`` returns (log amplitude, sign); ``forward`` returns Psi.
    Both interaction choices share orbitals, envelopes, determinants and factors.
    """

    def __init__(
        self,
        nuclear_coords,
        nuclear_charges,
        n_up,
        n_down,
        num_determinants=16,
        hidden_dim_one=64,
        hidden_dim_two=16,
        num_layers=2,
        eps=1e-24,
        use_cusp_jastrow=False,
        jastrow=None,
        jastrow_kwargs=None,
        spin_adapted=False,
        spin_projector=None,
        spin_projector_kwargs=None,
        envelope_type=None,
        smooth_cuspless=None,
        interaction_type="ferminet",
        envelope_scale=2.0,
    ):
        super().__init__()
        if min(n_up, n_down) < 0 or n_up + n_down == 0:
            raise ValueError("Electron counts must be nonnegative and nonempty")
        if min(num_layers, num_determinants, hidden_dim_one, hidden_dim_two) < 1:
            raise ValueError("Model dimensions and depth must be positive")
        if interaction_type not in ("ferminet", "psiformer"):
            raise ValueError("interaction_type must be ferminet or psiformer")
        self.n_up, self.n_down = n_up, n_down
        self.total_electrons = n_up + n_down
        self.num_determinants, self.num_layers = num_determinants, num_layers
        self.hidden_dim_one, self.hidden_dim_two = hidden_dim_one, hidden_dim_two
        self.eps, self.interaction_type = eps, interaction_type
        self.envelope_scale = float(envelope_scale)
        self.num_nuclei = len(nuclear_coords)
        self.register_buffer("nuclear_coords", nuclear_coords.detach().clone())
        self.register_buffer("nuclear_charges", nuclear_charges.detach().clone())
        self.envelope_type, self.smooth_cuspless = cusp_defaults(
            envelope_type,
            smooth_cuspless,
            jastrow is not None or use_cusp_jastrow,
            jastrow,
        )
        self.embedding = ElectronFeatureEmbedding(
            eps=eps, smooth_cuspless=self.smooth_cuspless
        )
        d1, d2 = self.embedding.feature_dims(self.num_nuclei, use_nuclear_charges=True)
        if interaction_type == "psiformer":
            self.input_proj = nn.Linear(d1, hidden_dim_one)
            self.layers = nn.ModuleList(
                [
                    PsiformerAttentionBlock(
                        hidden_dim_one,
                        max(1, hidden_dim_one // 16),
                        n_up=n_up,
                        n_down=n_down,
                    )
                    for _ in range(num_layers)
                ]
            )
        else:
            self.layers = nn.ModuleList(
                [
                    FermiInteractionLayer(
                        d1 if i == 0 else hidden_dim_one,
                        d2 if i == 0 else hidden_dim_two,
                        hidden_dim_one,
                        hidden_dim_two,
                        n_up,
                        n_down,
                        produce_two_electron_output=i != num_layers - 1,
                    )
                    for i in range(num_layers)
                ]
            )
        for name, count in (("up", n_up), ("down", n_down)):
            linear = (
                nn.Linear(hidden_dim_one, num_determinants * count) if count else None
            )
            env = (
                NuclearEnvelope(
                    num_determinants,
                    count,
                    self.num_nuclei,
                    envelope_type=self.envelope_type,
                    nuclear_charges=nuclear_charges,
                    envelope_scale=self.envelope_scale,
                )
                if count
                else None
            )
            setattr(self, "linear_orb_" + name, linear)
            setattr(self, "envelope_" + name, env)
        self.det_weights = nn.Parameter(
            torch.ones(num_determinants) / num_determinants**0.5
        )
        self.spin_projector = resolve_spin(
            n_up, n_down, spin_adapted, spin_projector, spin_projector_kwargs
        )
        pair_spin = pair_spin_for(n_up, n_down, self.spin_projector)
        if jastrow is None and use_cusp_jastrow:
            jk = dict(jastrow_kwargs or {})
            if pair_spin is not None:
                jk.setdefault("two_electron_spin", pair_spin)
            jastrow = KatoCuspJastrow(nuclear_charges, n_up, n_down, eps=eps, **jk)
        self.jastrow = jastrow
        validate_pair_cusp(jastrow, pair_spin)

    def _context(self, r, geom_ctx):
        return (
            geom_ctx
            if geom_ctx is not None and geom_ctx.valid_for(r, self.eps)
            else GeometricContext.compute(r, self.nuclear_coords, eps=self.eps)
        )

    def forward_orbitals(self, r, geom_ctx=None):
        if r.shape[1:] != (self.total_electrons, 3):
            raise ValueError("Coordinates must have shape (batch, total_electrons, 3)")
        ctx = self._context(r, geom_ctx)
        h1, h2 = self.embedding(
            r, self.nuclear_coords, self.nuclear_charges, geom_ctx=ctx
        )
        if self.interaction_type == "psiformer":
            h1 = self.input_proj(h1)
            for block in self.layers:
                h1 = block(h1, r_ee=ctx.r_ee, d_ee=ctx.d_ee)
        else:
            for layer in self.layers:
                h1, h2 = layer(h1, h2)

        def orbitals(start, stop, count, linear, envelope):
            if count == 0:
                return None
            backflow = linear(h1[:, start:stop]).reshape(
                r.shape[0], count, self.num_determinants, count
            )
            return backflow.permute(0, 2, 1, 3) * envelope(
                ctx.d_en[:, start:stop, :, 0]
            )

        return (
            orbitals(0, self.n_up, self.n_up, self.linear_orb_up, self.envelope_up),
            orbitals(
                self.n_up,
                self.total_electrons,
                self.n_down,
                self.linear_orb_down,
                self.envelope_down,
            ),
        )

    def _forward_log_psi_base(self, r, geom_ctx=None):
        return signed_determinant_sum(
            *self.forward_orbitals(r, geom_ctx), self.det_weights
        )

    def _forward_log_psi_unprojected(self, r, geom_ctx=None):
        ctx = self._context(r, geom_ctx)
        log, sign = self._forward_log_psi_base(r, ctx)
        if self.jastrow is not None:
            log = log + self.jastrow(r, self.nuclear_coords, geom_ctx=ctx)
        return log, sign

    def forward_log_psi(self, r, geom_ctx=None):
        if self.spin_projector is not None:
            # Projection changes coordinates/batch shape; its expanded batch
            # owns its geometry and must not reuse the original coordinate graph.
            return self.spin_projector.project_log_psi(
                self._forward_log_psi_unprojected, r
            )
        return self._forward_log_psi_unprojected(r, geom_ctx)

    def forward(self, r):
        log, sign = self.forward_log_psi(r)
        node = (sign == 0) | torch.isneginf(log)
        safe = torch.where(node, torch.zeros_like(log), log)
        return torch.where(node, torch.zeros_like(log), sign * torch.exp(safe))

    def has_trainable_jastrow(self):
        return self.jastrow is not None and any(True for _ in self.jastrow.parameters())

    def freeze_jastrow(self):
        if self.jastrow is not None:
            for parameter in self.jastrow.parameters():
                parameter.requires_grad_(False)
                parameter.grad = None

    def unfreeze_jastrow(self):
        if self.jastrow is not None:
            self.jastrow.requires_grad_(True)

    def get_parameter_groups(self, lr, jastrow_lr=None):
        jastrow = list(self.jastrow.parameters()) if self.jastrow is not None else []
        identifiers = {id(p) for p in jastrow}
        groups = [
            {
                "params": [
                    p
                    for p in self.parameters()
                    if p.requires_grad and id(p) not in identifiers
                ],
                "lr": lr,
                "name": "backbone",
            }
        ]
        trainable = [p for p in jastrow if p.requires_grad]
        if trainable:
            groups.append(
                {
                    "params": trainable,
                    "lr": lr if jastrow_lr is None else jastrow_lr,
                    "name": "jastrow",
                }
            )
        return groups
