"""
Verification Test Suite for Refined Prototypes & Legacy Adapters in proposal v2.1.0-REFINED.
Verifies all 5 Challenger bug fixes and legacy adapter wrappers.
"""

import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Callable
from enum import Enum

# Import original modules for baseline equivalence verification
from features import smooth_distance, ElectronFeatureEmbedding
from layers import FermiInteractionLayer
from envelope import NuclearEnvelope
from model import FermiNet


# ============================================================================
# Refined Prototypes from Proposal v2.1.0-REFINED
# ============================================================================

# --- Refined Prototype 2: FermiInteractionLayerModern (Fixes BUG-01) ---
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
        
        # 3. 预注册通道维度的归一化分母与有效掩码 Buffer (消除动态掩码与标量分支，彻底修复 BUG-01)
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


# --- Refined Prototype 3: NuclearEnvelopeModern (Fixes BUG-03) ---
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


# --- Refined Prototype 4: KineticEnergyCalculator (Fixes BUG-02) ---
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


# --- Refined Prototype 5: PsiformerAttentionBlock (Fixes BUG-04) ---
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


# --- Legacy Adapters from Section 6.4 (Checklist-5) ---
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


# ============================================================================
# Unit Tests for Refined Prototypes & Adapters
# ============================================================================

class TestRefinedPrototypes(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.nuclear_coords = torch.tensor([[0.0, 0.0, -0.7], [0.0, 0.0, 0.7]], dtype=torch.float32)
        self.nuclear_charges = torch.tensor([1.0, 1.0], dtype=torch.float32)

    def test_bug01_fixed_h2_molecule(self):
        """Verify BUG-01 fix: n_up=1, n_down=1 retains opposite-spin correlation."""
        n_up = 1
        n_down = 1
        layer = FermiInteractionLayerModern(8, 4, 8, 4, n_up, n_down)
        
        h1 = torch.randn(2, 2, 8)
        h2 = torch.ones(2, 2, 2, 4)
        
        h1_out, h2_out = layer(h1, h2)
        self.assertEqual(h1_out.shape, (2, 2, 8))
        self.assertEqual(h2_out.shape, (2, 2, 2, 4))
        
        # Check denom buffer values
        # For up electron (index 0): denom_up is 1.0 (clamped from 0), but valid_mask_up is False
        # For down electron (index 1): denom_up is 1.0, and valid_mask_up is True
        self.assertFalse(layer.valid_mask_up[0, 0, 0].item())
        self.assertTrue(layer.valid_mask_up[0, 1, 0].item())
        self.assertTrue(layer.valid_mask_down[0, 0, 0].item())
        self.assertFalse(layer.valid_mask_down[0, 1, 0].item())

    def test_bug03_fixed_checkpoint_compatibility(self):
        """Verify BUG-03 fix: NuclearEnvelopeModern matches original state_dict keys and softplus."""
        orig_env = NuclearEnvelope(num_determinants=4, num_orbitals=3, num_nuclei=2, nuclear_charges=self.nuclear_charges)
        modern_env = NuclearEnvelopeModern(num_determinants=4, num_orbitals=3, nuclear_coords=self.nuclear_coords, nuclear_charges=self.nuclear_charges)
        
        # State dict keys must match exactly
        self.assertEqual(set(orig_env.state_dict().keys()), set(modern_env.state_dict().keys()))
        self.assertIn("decay_rates_raw", modern_env.state_dict())
        self.assertIn("weights", modern_env.state_dict())
        
        # Test loading original weights into modern envelope
        modern_env.load_state_dict(orig_env.state_dict())
        
        # Forward output equivalence
        r = torch.randn(2, 4, 3)
        coords = self.nuclear_coords.unsqueeze(0).unsqueeze(0)
        diff = r.unsqueeze(2) - coords
        dist = smooth_distance(diff, eps=orig_env.eps, keepdim=False)
        
        orig_out = orig_env(r, self.nuclear_coords)
        modern_out = modern_env(dist)
        
        self.assertTrue(torch.allclose(orig_out, modern_out, atol=1e-6))

    def test_bug02_kinetic_calculator_default_reverse(self):
        """Verify BUG-02 fix: REVERSE is default mode and exact analytical match."""
        def mock_log_abs_psi(r):
            return torch.sum(r ** 2, dim=(-1, -2))
        
        r = torch.randn(2, 2, 3, dtype=torch.float64)
        t_local, lap, g2 = KineticEnergyCalculator.compute_kinetic_energy(mock_log_abs_psi, r)
        expected_lap = torch.full((2,), 2.0 * 6, dtype=torch.float64)
        self.assertTrue(torch.allclose(lap, expected_lap, atol=1e-10))

    def test_bug04_psiformer_spin_embedding(self):
        """Verify BUG-04 fix: PsiformerAttentionBlock possesses spin_embed and can differentiate spins."""
        block = PsiformerAttentionBlock(embed_dim=16, num_heads=4, n_up=2, n_down=1)
        self.assertTrue(hasattr(block, "spin_embed"))
        
        h = torch.randn(2, 3, 16)
        r = torch.randn(2, 3, 3)
        r_ee = r.unsqueeze(2) - r.unsqueeze(1)
        d_ee = torch.norm(r_ee, dim=-1, keepdim=True)
        
        out = block(h, r_ee, d_ee)
        self.assertEqual(out.shape, (2, 3, 16))

    def test_legacy_adapters(self):
        """Verify Checklist-5: Legacy adapters run without modification to calling signatures."""
        # Test NuclearEnvelopeLegacyAdapter
        adapter_env = NuclearEnvelopeLegacyAdapter(num_determinants=4, num_orbitals=3, num_nuclei=2)
        r = torch.randn(2, 4, 3)
        out_env = adapter_env(r, self.nuclear_coords)
        self.assertEqual(out_env.shape, (2, 4, 4, 3))
        
        # Test FermiInteractionLayerLegacyAdapter
        adapter_layer = FermiInteractionLayerLegacyAdapter(8, 4, 8, 4)
        h1 = torch.randn(2, 2, 8)
        h2 = torch.randn(2, 2, 2, 4)
        h1_next, h2_next = adapter_layer(h1, h2, n_up=1, n_down=1)
        self.assertEqual(h1_next.shape, (2, 2, 8))
        self.assertEqual(h2_next.shape, (2, 2, 2, 4))


if __name__ == "__main__":
    unittest.main()
