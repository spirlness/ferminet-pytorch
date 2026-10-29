"""
Empirical Challenger Independent Reverification Test Suite.
Author: Empirical Challenger (critic & specialist)
Target: ARCHITECTURE_REDUNDANCY_REVIEW_AND_REFACTORING_PROPOSAL.md (v2.1.0-REFINED)
Verifies:
  1. BUG-01: FermiInteractionLayerModern opposite-spin pooling retention in n_up=1 systems & multi-spin edge cases.
  2. BUG-02: KineticEnergyCalculator truthful claims, REVERSE/HUTCHINSON defaults, and Laplacian exactness.
  3. BUG-03: NuclearEnvelopeModern realistic broadcast, decay_rates_raw checkpoint compatibility, and softplus alignment.
  4. BUG-04: PsiformerAttentionBlock spin embedding and spin-channel differentiation.
  5. Checklist-5: Legacy Adapter Wrappers in Section 6.4 (FermiInteractionLayerLegacyAdapter, NuclearEnvelopeLegacyAdapter, FermiNetLegacyAdapter).
"""

import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Callable
from enum import Enum

from features import smooth_distance
from layers import FermiInteractionLayer
from envelope import NuclearEnvelope
from model import FermiNet


# ============================================================================
# Prototypes Extracted Directly from Proposal v2.1.0-REFINED (Section 5 & 6.4)
# ============================================================================

class FermiInteractionLayerModern(nn.Module):
    def __init__(
        self,
        d_in_single: int,
        d_in_double: int,
        d_out_single: int,
        d_out_double: int,
        n_up: int,
        n_down: int,
        activation: nn.Module = nn.Tanh(),
    ) -> None:
        super().__init__()
        self.d_in_single = d_in_single
        self.d_in_double = d_in_double
        self.d_out_single = d_out_single
        self.d_out_double = d_out_double
        self.n_up = n_up
        self.n_down = n_down
        self.total_electrons = n_up + n_down
        self.act = activation
        
        self.proj_h1_self = nn.Linear(d_in_single, d_out_single)
        self.proj_h1_mean_up = nn.Linear(d_in_single, d_out_single, bias=False) if n_up > 0 else None
        self.proj_h1_mean_down = nn.Linear(d_in_single, d_out_single, bias=False) if n_down > 0 else None
        self.proj_h2_mean_up = nn.Linear(d_in_double, d_out_single, bias=False) if n_up > 0 else None
        self.proj_h2_mean_down = nn.Linear(d_in_double, d_out_single, bias=False) if n_down > 0 else None
        self.res_proj_single = (
            nn.Linear(d_in_single, d_out_single, bias=False)
            if d_in_single != d_out_single else nn.Identity()
        )
        
        self.linear_double = nn.Linear(d_in_double, d_out_double)
        self.res_proj_double = (
            nn.Linear(d_in_double, d_out_double, bias=False)
            if d_in_double != d_out_double else nn.Identity()
        )
        
        counts_up = torch.full((1, self.total_electrons, 1), float(n_up))
        valid_mask_up = torch.ones((1, self.total_electrons, 1), dtype=torch.bool)
        if n_up > 0:
            counts_up[:, :n_up, :] = float(n_up - 1)
            if n_up == 1:
                valid_mask_up[:, :n_up, :] = False
        else:
            valid_mask_up[:] = False
        self.register_buffer("denom_up", counts_up.clamp(min=1.0))
        self.register_buffer("valid_mask_up", valid_mask_up)

        counts_down = torch.full((1, self.total_electrons, 1), float(n_down))
        valid_mask_down = torch.ones((1, self.total_electrons, 1), dtype=torch.bool)
        if n_down > 0:
            counts_down[:, n_up:, :] = float(n_down - 1)
            if n_down == 1:
                valid_mask_down[:, n_up:, :] = False
        else:
            valid_mask_down[:] = False
        self.register_buffer("denom_down", counts_down.clamp(min=1.0))
        self.register_buffer("valid_mask_down", valid_mask_down)

    def forward(
        self,
        h1: torch.Tensor,
        h2: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, _ = h1.shape
        
        if self.n_up > 0:
            sum_h2_up = torch.sum(h2[:, :, :self.n_up, :], dim=2)
            diag_up = torch.arange(self.n_up, device=h2.device)
            sum_h2_up[:, diag_up, :] = sum_h2_up[:, diag_up, :] - h2[:, diag_up, diag_up, :]
            mean_h2_up = torch.where(
                self.valid_mask_up,
                sum_h2_up / self.denom_up,
                torch.zeros_like(sum_h2_up),
            )
        else:
            mean_h2_up = torch.zeros(B, N, self.d_in_double, device=h2.device, dtype=h2.dtype)
            
        if self.n_down > 0:
            sum_h2_down = torch.sum(h2[:, :, self.n_up:, :], dim=2)
            diag_down = torch.arange(self.n_down, device=h2.device) + self.n_up
            sum_h2_down[:, diag_down, :] = sum_h2_down[:, diag_down, :] - h2[:, diag_down, diag_down, :]
            mean_h2_down = torch.where(
                self.valid_mask_down,
                sum_h2_down / self.denom_down,
                torch.zeros_like(sum_h2_down),
            )
        else:
            mean_h2_down = torch.zeros(B, N, self.d_in_double, device=h2.device, dtype=h2.dtype)
            
        h1_out = self.proj_h1_self(h1)
        if self.proj_h1_mean_up is not None:
            mean_h1_up = torch.mean(h1[:, :self.n_up, :], dim=1, keepdim=True)
            h1_out = h1_out + self.proj_h1_mean_up(mean_h1_up)
            
        if self.proj_h1_mean_down is not None:
            mean_h1_down = torch.mean(h1[:, self.n_up:, :], dim=1, keepdim=True)
            h1_out = h1_out + self.proj_h1_mean_down(mean_h1_down)
            
        if self.proj_h2_mean_up is not None:
            h1_out = h1_out + self.proj_h2_mean_up(mean_h2_up)
            
        if self.proj_h2_mean_down is not None:
            h1_out = h1_out + self.proj_h2_mean_down(mean_h2_down)
            
        h1_next = self.act(h1_out) + self.res_proj_single(h1)
        h2_transformed = self.act(self.linear_double(h2))
        h2_next = h2_transformed + self.res_proj_double(h2)
        return h1_next, h2_next


class NuclearEnvelopeModern(nn.Module):
    def __init__(
        self,
        num_determinants: int,
        num_orbitals: int,
        nuclear_coords: torch.Tensor,
        init_decay_rate: float = 1.0,
        envelope_type: str = "exponential",
        nuclear_charges: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.num_determinants = num_determinants
        self.num_orbitals = num_orbitals
        self.num_nuclei = nuclear_coords.shape[0]
        self.envelope_type = envelope_type.lower()
        
        self.weights = nn.Parameter(
            torch.empty(num_determinants, num_orbitals, self.num_nuclei)
        )
        self.decay_rates_raw = nn.Parameter(
            torch.empty(num_determinants, num_orbitals, self.num_nuclei)
        )
        self._reset_parameters(init_decay_rate, nuclear_charges=nuclear_charges)

    def _reset_parameters(
        self,
        init_decay_rate: float,
        nuclear_charges: Optional[torch.Tensor] = None,
    ) -> None:
        nn.init.normal_(self.weights, mean=1.0, std=0.1)
        if nuclear_charges is not None:
            charges = nuclear_charges.to(
                device=self.decay_rates_raw.device, dtype=self.decay_rates_raw.dtype
            )
            charges = torch.clamp(charges, min=1e-4)
            inv_vals = torch.where(charges > 20.0, charges, torch.log(torch.expm1(charges)))
            with torch.no_grad():
                self.decay_rates_raw.data.copy_(
                    inv_vals.reshape(1, 1, self.num_nuclei).expand_as(self.decay_rates_raw)
                )
        else:
            inv_val = torch.log(torch.exp(torch.tensor(init_decay_rate)) - 1.0).item()
            nn.init.constant_(self.decay_rates_raw, inv_val)

    def forward(
        self,
        dist_en: torch.Tensor,
    ) -> torch.Tensor:
        if dist_en.dim() == 4:
            dist_en = dist_en.squeeze(-1)
            
        alpha = F.softplus(self.decay_rates_raw)
        
        B, N, M = dist_en.shape
        K, O, _ = alpha.shape
        
        dist_expanded = dist_en.unsqueeze(2).unsqueeze(3)
        alpha_expanded = alpha.view(1, 1, K, O, M)
        weights_expanded = self.weights.view(1, 1, K, O, M)
        
        if self.envelope_type == "gaussian":
            decay = torch.exp(-alpha_expanded * (dist_expanded ** 2))
        else:
            decay = torch.exp(-alpha_expanded * dist_expanded)
            
        envelope = torch.sum(weights_expanded * decay, dim=-1)
        return envelope.permute(0, 2, 1, 3).contiguous()


class KineticMode(str, Enum):
    REVERSE = "reverse"
    HUTCHINSON = "hutchinson"
    FORWARD_JVP = "forward_jvp"


class KineticEnergyCalculator:
    @staticmethod
    def compute_kinetic_energy(
        log_abs_psi_fn: Callable[[torch.Tensor], torch.Tensor],
        r: torch.Tensor,
        mode: KineticMode = KineticMode.REVERSE,
        num_probes: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, N, _ = r.shape
        dim_total = N * 3
        
        r_eval = r.detach().clone().requires_grad_(True)
        
        if mode == KineticMode.HUTCHINSON:
            log_psi = log_abs_psi_fn(r_eval)
            g = torch.autograd.grad(log_psi.sum(), r_eval, create_graph=True)[0]
            grad_squared = torch.sum(g.reshape(B, -1) ** 2, dim=-1)
            
            laplacian = torch.zeros(B, device=r.device, dtype=r.dtype)
            for _ in range(num_probes):
                v = torch.randint(0, 2, g.shape, device=r.device, dtype=r.dtype) * 2.0 - 1.0
                vg = torch.sum(v * g)
                hvp = torch.autograd.grad(vg, r_eval, retain_graph=True)[0]
                laplacian = laplacian + torch.sum(v * hvp, dim=(-1, -2))
                
            laplacian = laplacian / float(num_probes)
            t_local = -0.5 * (laplacian + grad_squared)
            return t_local, laplacian, grad_squared

        elif mode == KineticMode.FORWARD_JVP:
            log_psi = log_abs_psi_fn(r_eval)
            g = torch.autograd.grad(log_psi.sum(), r_eval, create_graph=True)[0]
            grad_flat = g.reshape(B, dim_total)
            grad_squared = torch.sum(grad_flat ** 2, dim=-1)
            
            laplacian = torch.zeros(B, device=r.device, dtype=r.dtype)
            eye = torch.eye(dim_total, device=r.device, dtype=r.dtype)
            
            for d in range(dim_total):
                v_d = eye[d].unsqueeze(0).expand(B, -1).reshape(B, N, 3)
                with torch.autograd.forward_ad.dual_level():
                    dual_r = torch.autograd.forward_ad.make_dual(r_eval, v_d)
                    dual_log_psi = log_abs_psi_fn(dual_r)
                    jvp_grad = torch.autograd.forward_ad.unpack_dual(dual_log_psi).tangent
                    sec_retain = (d < dim_total - 1)
                    sec_g = torch.autograd.grad(jvp_grad.sum(), r_eval, retain_graph=sec_retain)[0]
                    laplacian = laplacian + sec_g.reshape(B, dim_total)[:, d]
                    
            t_local = -0.5 * (laplacian + grad_squared)
            return t_local, laplacian, grad_squared

        else:
            log_psi = log_abs_psi_fn(r_eval)
            ones = torch.ones_like(log_psi)
            g = torch.autograd.grad(log_psi, r_eval, grad_outputs=ones, create_graph=True)[0]
            grad_flat = g.reshape(B, dim_total)
            grad_squared = torch.sum(grad_flat ** 2, dim=-1)
            
            laplacian = torch.zeros(B, device=r.device, dtype=r.dtype)
            for d in range(dim_total):
                g_d = grad_flat[:, d]
                sec_retain = (d < dim_total - 1)
                sec_grad = torch.autograd.grad(
                    outputs=g_d,
                    inputs=r_eval,
                    grad_outputs=ones,
                    create_graph=False,
                    retain_graph=sec_retain,
                )[0]
                laplacian = laplacian + sec_grad.reshape(B, dim_total)[:, d]
                
            t_local = -0.5 * (laplacian + grad_squared)
            return t_local, laplacian, grad_squared


class PsiformerAttentionBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        mlp_ratio: int = 2,
        n_up: Optional[int] = None,
        n_down: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = 1.0 / (self.head_dim ** 0.5)
        self.n_up = n_up
        self.n_down = n_down
        
        self.spin_embed = nn.Embedding(2, embed_dim)
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        
        self.geom_bias_mlp = nn.Sequential(
            nn.Linear(4, embed_dim // 2),
            nn.Tanh(),
            nn.Linear(embed_dim // 2, num_heads),
        )
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, mlp_ratio * embed_dim),
            nn.GELU(),
            nn.Linear(mlp_ratio * embed_dim, embed_dim),
        )

    def forward(
        self,
        h: torch.Tensor,
        r_ee: Optional[torch.Tensor] = None,
        d_ee: Optional[torch.Tensor] = None,
        spin_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, N, C = h.shape
        
        if spin_ids is not None:
            h_input = h + self.spin_embed(spin_ids)
        elif self.n_up is not None and self.n_down is not None:
            s_idx = torch.zeros(N, dtype=torch.long, device=h.device)
            s_idx[self.n_up:] = 1
            h_input = h + self.spin_embed(s_idx).unsqueeze(0)
        else:
            h_input = h
        
        norm_h = self.norm1(h_input)
        qkv = self.qkv_proj(norm_h).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        
        attn_scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        
        if r_ee is not None and d_ee is not None:
            geom_feat = torch.cat([r_ee, d_ee], dim=-1)
            geom_bias = self.geom_bias_mlp(geom_feat).permute(0, 3, 1, 2)
            attn_scores = attn_scores + geom_bias
            
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_out = torch.matmul(attn_weights, v)
        attn_out = attn_out.permute(0, 2, 1, 3).reshape(B, N, C)
        
        h = h + self.out_proj(attn_out)
        h = h + self.ffn(self.norm2(h))
        return h


# Legacy Adapters (Section 6.4)
class NuclearEnvelopeLegacyAdapter(nn.Module):
    def __init__(
        self,
        num_determinants: int,
        num_orbitals: int,
        num_nuclei: int,
        eps: float = 1e-24,
        init_decay_rate: float = 1.0,
        envelope_type: str = "exponential",
        nuclear_charges: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.eps = eps
        dummy_coords = torch.zeros(num_nuclei, 3)
        self.modern = NuclearEnvelopeModern(
            num_determinants=num_determinants,
            num_orbitals=num_orbitals,
            nuclear_coords=dummy_coords,
            init_decay_rate=init_decay_rate,
            envelope_type=envelope_type,
            nuclear_charges=nuclear_charges,
        )
        self.weights = self.modern.weights
        self.decay_rates_raw = self.modern.decay_rates_raw
        self.envelope_type = self.modern.envelope_type

    def forward(
        self,
        r: torch.Tensor,
        nuclear_coords: torch.Tensor,
    ) -> torch.Tensor:
        coords = nuclear_coords.to(device=r.device, dtype=r.dtype)
        diff = r.unsqueeze(2) - coords.unsqueeze(0).unsqueeze(0)
        dist = smooth_distance(diff, eps=self.eps, keepdim=False)
        return self.modern(dist)


class FermiInteractionLayerLegacyAdapter(nn.Module):
    def __init__(
        self,
        in_dim_one: int,
        in_dim_two: int,
        out_dim_one: int,
        out_dim_two: int,
        produce_two_electron_output: bool = True,
        activation: nn.Module = nn.Tanh(),
    ) -> None:
        super().__init__()
        self.in_dim_one = in_dim_one
        self.in_dim_two = in_dim_two
        self.out_dim_one = out_dim_one
        self.out_dim_two = out_dim_two
        self.produce_two_electron_output = produce_two_electron_output
        self.act = activation
        self._modern_layers = nn.ModuleDict()

    def _get_modern_layer(self, n_up: int, n_down: int) -> nn.Module:
        key = f"modern_{n_up}_{n_down}"
        if key not in self._modern_layers:
            self._modern_layers[key] = FermiInteractionLayerModern(
                d_in_single=self.in_dim_one,
                d_in_double=self.in_dim_two,
                d_out_single=self.out_dim_one,
                d_out_double=self.out_dim_two,
                n_up=n_up,
                n_down=n_down,
                activation=self.act,
            )
        return self._modern_layers[key]

    def forward(
        self,
        h1: torch.Tensor,
        h2: torch.Tensor,
        n_up: int,
        n_down: int,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        modern_layer = self._get_modern_layer(n_up, n_down)
        h1_next, h2_next = modern_layer(h1, h2)
        if not self.produce_two_electron_output:
            h2_next = None
        return h1_next, h2_next


class FermiNetLegacyAdapter(nn.Module):
    def __init__(
        self,
        nuclear_coords: torch.Tensor,
        nuclear_charges: torch.Tensor,
        n_up: int,
        n_down: int,
        num_determinants: int = 16,
        hidden_dim_one: int = 256,
        hidden_dim_two: int = 32,
        num_layers: int = 4,
        eps: float = 1e-6,
        use_cusp_jastrow: bool = False,
        jastrow: Optional[nn.Module] = None,
        jastrow_kwargs: Optional[dict] = None,
        spin_adapted: bool = False,
        spin_projector: Optional[nn.Module] = None,
        spin_projector_kwargs: Optional[dict] = None,
        envelope_type: Optional[str] = None,
        smooth_cuspless: Optional[bool] = None,
        interaction_type: str = "ferminet",
    ) -> None:
        super().__init__()
        self.n_up = n_up
        self.n_down = n_down
        self.total_electrons = n_up + n_down
        self.num_determinants = num_determinants
        self.interaction_type = interaction_type
        
        self.register_buffer("nuclear_coords", nuclear_coords.clone().detach())
        self.register_buffer("nuclear_charges", nuclear_charges.clone().detach())
        
        from model import FermiNet
        self._delegate = FermiNet(
            nuclear_coords=nuclear_coords,
            nuclear_charges=nuclear_charges,
            n_up=n_up,
            n_down=n_down,
            num_determinants=num_determinants,
            hidden_dim_one=hidden_dim_one,
            hidden_dim_two=hidden_dim_two,
            num_layers=num_layers,
            eps=eps,
            use_cusp_jastrow=use_cusp_jastrow,
            jastrow=jastrow,
            jastrow_kwargs=jastrow_kwargs,
            spin_adapted=spin_adapted,
            spin_projector=spin_projector,
            spin_projector_kwargs=spin_projector_kwargs,
            envelope_type=envelope_type,
            smooth_cuspless=smooth_cuspless,
        )

    def forward_log_psi(self, r: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._delegate.forward_log_psi(r)

    def forward(self, r: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.forward_log_psi(r)


# ============================================================================
# Empirical Adversarial Test Cases
# ============================================================================

class TestEmpiricalChallengerReverification(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(12345)
        self.coords_h2 = torch.tensor([[0.0, 0.0, -0.7], [0.0, 0.0, 0.7]], dtype=torch.float32)
        self.charges_h2 = torch.tensor([1.0, 1.0], dtype=torch.float32)

    # ------------------------------------------------------------------------
    # 1. BUG-01 Tests: FermiInteractionLayerModern
    # ------------------------------------------------------------------------
    def test_bug01_adversarial_spin_pooling_mathematical_oracle(self):
        """
        Adversarial stress-test:
        Compare manual ground truth pooling against FermiInteractionLayerModern for:
          - n_up=1, n_down=1 (H2, critical case)
          - n_up=2, n_down=1 (Li, spin asymmetric)
          - n_up=3, n_down=3 (C-like doublet/triplet)
          - n_up=1, n_down=0 (Hydrogen atom, extreme edge case)
        """
        cases = [
            (1, 1),
            (2, 1),
            (3, 3),
            (1, 0),
        ]
        
        for n_up, n_down in cases:
            N = n_up + n_down
            B = 3
            D2 = 5
            D1 = 8
            
            layer = FermiInteractionLayerModern(
                d_in_single=D1,
                d_in_double=D2,
                d_out_single=D1,
                d_out_double=D2,
                n_up=n_up,
                n_down=n_down,
            )
            
            # Generate deterministic non-zero h2
            h2 = torch.randn(B, N, N, D2)
            # Make sure diagonals are distinct non-zero
            for i in range(N):
                h2[:, i, i, :] = torch.full((B, D2), float(i + 10.0))
            
            # Compute ground-truth pooling manually
            expected_mean_up = torch.zeros(B, N, D2)
            expected_mean_down = torch.zeros(B, N, D2)
            
            for i in range(N):
                # up pooling for electron i
                if n_up > 0:
                    up_indices = [j for j in range(n_up) if j != i]
                    if len(up_indices) > 0:
                        expected_mean_up[:, i, :] = torch.mean(h2[:, i, up_indices, :], dim=1)
                    else:
                        expected_mean_up[:, i, :] = 0.0
                
                # down pooling for electron i
                if n_down > 0:
                    down_indices = [j for j in range(n_up, N) if j != i]
                    if len(down_indices) > 0:
                        expected_mean_down[:, i, :] = torch.mean(h2[:, i, down_indices, :], dim=1)
                    else:
                        expected_mean_down[:, i, :] = 0.0
            
            # Now extract pooling inside forward
            # sum_h2_up
            if n_up > 0:
                sum_up = torch.sum(h2[:, :, :n_up, :], dim=2)
                diag_up = torch.arange(n_up)
                sum_up[:, diag_up, :] -= h2[:, diag_up, diag_up, :]
                modern_mean_up = torch.where(
                    layer.valid_mask_up,
                    sum_up / layer.denom_up,
                    torch.zeros_like(sum_up),
                )
            else:
                modern_mean_up = torch.zeros(B, N, D2)
                
            if n_down > 0:
                sum_down = torch.sum(h2[:, :, n_up:, :], dim=2)
                diag_down = torch.arange(n_down) + n_up
                sum_down[:, diag_down, :] -= h2[:, diag_down, diag_down, :]
                modern_mean_down = torch.where(
                    layer.valid_mask_down,
                    sum_down / layer.denom_down,
                    torch.zeros_like(sum_down),
                )
            else:
                modern_mean_down = torch.zeros(B, N, D2)
            
            # Check BUG-01 key property:
            if n_up == 1 and n_down == 1:
                # Electron 1 (down) seeing up: must equal h2[:, 1, 0, :]
                self.assertTrue(torch.allclose(modern_mean_up[:, 1, :], h2[:, 1, 0, :], atol=1e-6))
                # Electron 0 (up) seeing up: must be exactly 0.0
                self.assertTrue(torch.allclose(modern_mean_up[:, 0, :], torch.zeros_like(modern_mean_up[:, 0, :]), atol=1e-6))
                # Electron 0 (up) seeing down: must equal h2[:, 0, 1, :]
                self.assertTrue(torch.allclose(modern_mean_down[:, 0, :], h2[:, 0, 1, :], atol=1e-6))
                # Electron 1 (down) seeing down: must be exactly 0.0
                self.assertTrue(torch.allclose(modern_mean_down[:, 1, :], torch.zeros_like(modern_mean_down[:, 1, :]), atol=1e-6))
                
            # Full mathematical oracle check across all electrons
            diff_up = torch.max(torch.abs(modern_mean_up - expected_mean_up)).item()
            diff_down = torch.max(torch.abs(modern_mean_down - expected_mean_down)).item()
            self.assertLess(diff_up, 1e-6, f"Mismatch in mean_up for ({n_up}, {n_down}): diff={diff_up}")
            self.assertLess(diff_down, 1e-6, f"Mismatch in mean_down for ({n_up}, {n_down}): diff={diff_down}")

    def test_bug01_autograd_and_finiteness(self):
        """Verify autograd gradient flow and finiteness for n_up=1, n_down=1."""
        layer = FermiInteractionLayerModern(8, 4, 16, 8, n_up=1, n_down=1)
        h1 = torch.randn(4, 2, 8, requires_grad=True)
        h2 = torch.randn(4, 2, 2, 4, requires_grad=True)
        
        h1_out, h2_out = layer(h1, h2)
        loss = h1_out.sum() + h2_out.sum()
        loss.backward()
        
        self.assertIsNotNone(h1.grad)
        self.assertIsNotNone(h2.grad)
        self.assertTrue(torch.isfinite(h1.grad).all())
        self.assertTrue(torch.isfinite(h2.grad).all())

    # ------------------------------------------------------------------------
    # 2. BUG-02 Tests: KineticEnergyCalculator
    # ------------------------------------------------------------------------
    def test_bug02_kinetic_calculator_reverse_exact_harmonic(self):
        """
        Verify that REVERSE mode gives exact analytical Laplacian and kinetic energy:
        Test wavefunction: psi = exp(-0.5 * alpha * sum(r^2))
        log_psi = -0.5 * alpha * sum(r^2)
        grad(log_psi) = -alpha * r
        laplacian = -alpha * 3 * N
        grad_squared = alpha^2 * sum(r^2)
        T_L = -0.5 * (laplacian + grad_squared) = 0.5 * (3*alpha*N - alpha^2*sum(r^2))
        """
        alpha = 1.75
        N = 4
        B = 2
        r = torch.randn(B, N, 3, dtype=torch.float64, requires_grad=True)
        
        def log_abs_psi(pos):
            return -0.5 * alpha * torch.sum(pos ** 2, dim=(-1, -2))
        
        t_rev, lap_rev, g2_rev = KineticEnergyCalculator.compute_kinetic_energy(
            log_abs_psi, r, mode=KineticMode.REVERSE
        )
        
        expected_lap = torch.full((B,), -alpha * 3 * N, dtype=torch.float64)
        expected_g2 = (alpha ** 2) * torch.sum(r ** 2, dim=(-1, -2))
        expected_t = -0.5 * (expected_lap + expected_g2)
        
        self.assertTrue(torch.allclose(lap_rev, expected_lap, atol=1e-10))
        self.assertTrue(torch.allclose(g2_rev, expected_g2, atol=1e-10))
        self.assertTrue(torch.allclose(t_rev, expected_t, atol=1e-10))

    def test_bug02_kinetic_calculator_hutchinson_unbiasedness(self):
        """
        Verify that Hutchinson random trace estimator is statistically unbiased:
        E[lap_hutchinson] == lap_exact
        """
        alpha = 1.0
        N = 3
        B = 1
        r = torch.randn(B, N, 3, dtype=torch.float64)
        
        def log_abs_psi(pos):
            return -0.5 * alpha * torch.sum(pos ** 2, dim=(-1, -2))
            
        _, lap_exact, _ = KineticEnergyCalculator.compute_kinetic_energy(
            log_abs_psi, r, mode=KineticMode.REVERSE
        )
        
        # Run Hutchinson with 100 probes
        _, lap_hutch, _ = KineticEnergyCalculator.compute_kinetic_energy(
            log_abs_psi, r, mode=KineticMode.HUTCHINSON, num_probes=100
        )
        
        # Monte Carlo error with 100 probes for diagonal Hessian of quadratic should be small
        self.assertTrue(torch.allclose(lap_hutch, lap_exact, atol=0.5))

    def test_bug02_kinetic_calculator_forward_jvp_correctness(self):
        """Verify that FORWARD_JVP mode calculates correct Laplacian matching REVERSE mode."""
        alpha = 1.2
        N = 2
        B = 2
        r = torch.randn(B, N, 3, dtype=torch.float64)
        
        def log_abs_psi(pos):
            return -0.5 * alpha * torch.sum(pos ** 2, dim=(-1, -2))
            
        t_rev, lap_rev, g2_rev = KineticEnergyCalculator.compute_kinetic_energy(
            log_abs_psi, r, mode=KineticMode.REVERSE
        )
        t_jvp, lap_jvp, g2_jvp = KineticEnergyCalculator.compute_kinetic_energy(
            log_abs_psi, r, mode=KineticMode.FORWARD_JVP
        )
        
        self.assertTrue(torch.allclose(lap_rev, lap_jvp, atol=1e-8))
        self.assertTrue(torch.allclose(t_rev, t_jvp, atol=1e-8))

    # ------------------------------------------------------------------------
    # 3. BUG-03 Tests: NuclearEnvelopeModern
    # ------------------------------------------------------------------------
    def test_bug03_nuclear_envelope_softplus_and_charge_alignment(self):
        """Verify softplus regularization and nuclear charge physics alignment."""
        charges = torch.tensor([1.0, 6.0, 8.0], dtype=torch.float32)
        coords = torch.randn(3, 3)
        env = NuclearEnvelopeModern(
            num_determinants=4,
            num_orbitals=5,
            nuclear_coords=coords,
            nuclear_charges=charges,
        )
        
        # Check that alpha = softplus(decay_rates_raw) matches charges
        alpha = F.softplus(env.decay_rates_raw)
        # alpha shape: (4, 5, 3)
        for i in range(3):
            expected_charge = charges[i].item()
            actual_decay = alpha[:, :, i]
            self.assertTrue(
                torch.allclose(actual_decay, torch.full_like(actual_decay, expected_charge), atol=1e-5),
                f"Decay rate mismatch for nucleus {i}: expected {expected_charge}, got {actual_decay[0, 0].item()}"
            )

    def test_bug03_nuclear_envelope_checkpoint_bidirectional_state_dict(self):
        """Verify 100% parameter compatibility between NuclearEnvelope and NuclearEnvelopeModern."""
        coords = torch.tensor([[0.0, 0.0, -1.0], [0.0, 0.0, 1.0]])
        charges = torch.tensor([1.0, 1.0])
        
        orig_env = NuclearEnvelope(num_determinants=8, num_orbitals=6, num_nuclei=2, nuclear_charges=charges)
        modern_env = NuclearEnvelopeModern(num_determinants=8, num_orbitals=6, nuclear_coords=coords, nuclear_charges=charges)
        
        # Perturb original parameters to non-trivial values
        with torch.no_grad():
            orig_env.weights.normal_(0.5, 0.2)
            orig_env.decay_rates_raw.normal_(1.2, 0.3)
            
        # Migrate state dict
        modern_env.load_state_dict(orig_env.state_dict())
        
        # Test outputs on random electron positions
        r = torch.randn(5, 4, 3)
        diff = r.unsqueeze(2) - coords.unsqueeze(0).unsqueeze(0)
        dist = smooth_distance(diff, eps=orig_env.eps, keepdim=False)
        
        out_orig = orig_env(r, coords)
        out_modern = modern_env(dist)
        
        self.assertEqual(out_orig.shape, out_modern.shape)
        self.assertTrue(torch.allclose(out_orig, out_modern, atol=1e-6))

    # ------------------------------------------------------------------------
    # 4. BUG-04 Tests: PsiformerAttentionBlock
    # ------------------------------------------------------------------------
    def test_bug04_psiformer_spin_differentiation(self):
        """
        Verify that PsiformerAttentionBlock breaks spin degeneracy:
        If two electrons have identical spatial coordinates but opposite spins,
        their output representations MUST differ due to spin_embed.
        """
        block = PsiformerAttentionBlock(embed_dim=16, num_heads=2, n_up=1, n_down=1)
        
        # Electron 0 and Electron 1 have identical h input
        h = torch.ones(1, 2, 16)
        r_ee = torch.zeros(1, 2, 2, 3)
        d_ee = torch.zeros(1, 2, 2, 1)
        
        out = block(h, r_ee, d_ee)
        
        out_up = out[0, 0, :]
        out_down = out[0, 1, :]
        
        diff = torch.norm(out_up - out_down).item()
        self.assertGreater(
            diff, 1e-3, "Psiformer failed to differentiate opposite-spin electrons with identical positions!"
        )

    def test_bug04_psiformer_same_spin_permutation_equivariance(self):
        """
        Verify permutation equivariance among electrons with the same spin:
        Permuting two up electrons must permute their outputs identically.
        """
        block = PsiformerAttentionBlock(embed_dim=16, num_heads=2, n_up=2, n_down=0)
        
        r1 = torch.tensor([0.1, 0.2, 0.3])
        r2 = torch.tensor([0.4, 0.5, 0.6])
        h1 = torch.randn(1, 16)
        h2 = torch.randn(1, 16)
        
        h_a = torch.stack([h1, h2], dim=1) # (1, 2, 16)
        r_a = torch.stack([r1, r2], dim=0).unsqueeze(0) # (1, 2, 3)
        r_ee_a = r_a.unsqueeze(2) - r_a.unsqueeze(1)
        d_ee_a = torch.norm(r_ee_a, dim=-1, keepdim=True)
        
        # Permuted
        h_b = torch.stack([h2, h1], dim=1)
        r_b = torch.stack([r2, r1], dim=0).unsqueeze(0)
        r_ee_b = r_b.unsqueeze(2) - r_b.unsqueeze(1)
        d_ee_b = torch.norm(r_ee_b, dim=-1, keepdim=True)
        
        out_a = block(h_a, r_ee_a, d_ee_a)
        out_b = block(h_b, r_ee_b, d_ee_b)
        
        # Out_b[0, 0] should equal Out_a[0, 1], and Out_b[0, 1] should equal Out_a[0, 0]
        self.assertTrue(torch.allclose(out_a[0, 0], out_b[0, 1], atol=1e-5))
        self.assertTrue(torch.allclose(out_a[0, 1], out_b[0, 0], atol=1e-5))

    # ------------------------------------------------------------------------
    # 5. Checklist-5 Tests: Section 6.4 Legacy Adapters
    # ------------------------------------------------------------------------
    def test_checklist5_ferminet_legacy_adapter_end_to_end(self):
        """Verify FermiNetLegacyAdapter forward and forward_log_psi equivalence."""
        model = FermiNetLegacyAdapter(
            nuclear_coords=self.coords_h2,
            nuclear_charges=self.charges_h2,
            n_up=1,
            n_down=1,
            num_determinants=4,
            hidden_dim_one=32,
            hidden_dim_two=16,
            num_layers=2,
        )
        
        r = torch.randn(3, 2, 3)
        log_psi, sign = model.forward_log_psi(r)
        self.assertEqual(log_psi.shape, (3,))
        self.assertEqual(sign.shape, (3,))
        self.assertTrue(torch.isfinite(log_psi).all())
        
        psi = model(r)
        self.assertEqual(psi[0].shape, (3,))

    def test_checklist5_interaction_legacy_adapter_delegation(self):
        """Verify FermiInteractionLayerLegacyAdapter matches original layer interface."""
        adapter = FermiInteractionLayerLegacyAdapter(
            in_dim_one=16,
            in_dim_two=8,
            out_dim_one=32,
            out_dim_two=16,
            produce_two_electron_output=True,
        )
        
        h1 = torch.randn(2, 3, 16)
        h2 = torch.randn(2, 3, 3, 8)
        
        # Call with n_up=2, n_down=1
        h1_out, h2_out = adapter(h1, h2, n_up=2, n_down=1)
        self.assertEqual(h1_out.shape, (2, 3, 32))
        self.assertEqual(h2_out.shape, (2, 3, 3, 16))
        
        # Test produce_two_electron_output=False
        adapter_no_two = FermiInteractionLayerLegacyAdapter(
            in_dim_one=16, in_dim_two=8, out_dim_one=32, out_dim_two=16, produce_two_electron_output=False
        )
        h1_only, h2_none = adapter_no_two(h1, h2, n_up=2, n_down=1)
        self.assertIsNone(h2_none)
        self.assertEqual(h1_only.shape, (2, 3, 32))


if __name__ == "__main__":
    unittest.main()
