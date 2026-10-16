"""
Empirical Verification and Challenge Test Suite for Chapter 5 Modernization Prototypes.
Tests:
1. GeometricContext
2. FermiInteractionLayerModern
3. NuclearEnvelopeModern
4. KineticEnergyCalculator
5. PsiformerAttentionBlock
"""

import unittest
import math
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import existing implementations for baseline comparison
from features import smooth_distance, ElectronFeatureEmbedding
from layers import FermiInteractionLayer
from envelope import NuclearEnvelope


# ============================================================================
# Prototypes from Chapter 5 of the Proposal
# ============================================================================

# --- Prototype 1: GeometricContext ---
@dataclass(frozen=True)
class GeometricContext:
    r: torch.Tensor
    r_en: torch.Tensor
    d_en: torch.Tensor
    r_ee: Optional[torch.Tensor] = None
    d_ee: Optional[torch.Tensor] = None

    @classmethod
    def compute(
        cls,
        r: torch.Tensor,
        nuclear_coords: torch.Tensor,
        eps: float = 1e-12,
        compute_ee: bool = True,
    ) -> "GeometricContext":
        batch_size, num_electrons, _ = r.shape
        num_nuclei, _ = nuclear_coords.shape
        
        r_expanded = r.unsqueeze(2)  # (B, N, 1, 3)
        nuc_expanded = nuclear_coords.unsqueeze(0).unsqueeze(0)  # (1, 1, M, 3)
        r_en = r_expanded - nuc_expanded  # (B, N, M, 3)
        d_en = torch.sqrt(torch.sum(r_en ** 2, dim=-1, keepdim=True) + eps)  # (B, N, M, 1)
        
        r_ee: Optional[torch.Tensor] = None
        d_ee: Optional[torch.Tensor] = None
        if compute_ee and num_electrons > 1:
            r_i = r.unsqueeze(2)  # (B, N, 1, 3)
            r_j = r.unsqueeze(1)  # (B, 1, N, 3)
            r_ee = r_i - r_j      # (B, N, N, 3)
            d_ee = torch.sqrt(torch.sum(r_ee ** 2, dim=-1, keepdim=True) + eps)  # (B, N, N, 1)
            
        return cls(r=r, r_en=r_en, d_en=d_en, r_ee=r_ee, d_ee=d_ee)


# --- Prototype 2: FermiInteractionLayerModern ---
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
        
        denom_up = torch.full((1, self.total_electrons, 1), max(float(n_up), 1.0))
        if n_up > 0:
            denom_up[:, :n_up, :] = max(float(n_up - 1), 1.0)
        self.register_buffer("denom_up", denom_up)

        denom_down = torch.full((1, self.total_electrons, 1), max(float(n_down), 1.0))
        if n_down > 0:
            denom_down[:, n_up:, :] = max(float(n_down - 1), 1.0)
        self.register_buffer("denom_down", denom_down)

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
            mean_h2_up = sum_h2_up / self.denom_up if self.n_up > 1 else torch.zeros_like(sum_h2_up)
        else:
            mean_h2_up = torch.zeros(B, N, self.d_in_double, device=h2.device, dtype=h2.dtype)
            
        if self.n_down > 0:
            sum_h2_down = torch.sum(h2[:, :, self.n_up:, :], dim=2)
            diag_down = torch.arange(self.n_down, device=h2.device) + self.n_up
            sum_h2_down[:, diag_down, :] = sum_h2_down[:, diag_down, :] - h2[:, diag_down, diag_down, :]
            mean_h2_down = sum_h2_down / self.denom_down if self.n_down > 1 else torch.zeros_like(sum_h2_down)
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


# --- Prototype 3: NuclearEnvelopeModern ---
class NuclearEnvelopeModern(nn.Module):
    def __init__(
        self,
        num_determinants: int,
        num_orbitals: int,
        nuclear_coords: torch.Tensor,
        init_alpha: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_k = num_determinants
        self.num_orb = num_orbitals
        self.num_nuclei = nuclear_coords.shape[0]
        
        self.log_alpha = nn.Parameter(
            torch.full((self.num_k, self.num_orb, self.num_nuclei), float(torch.log(torch.tensor(init_alpha))))
        )
        self.weights = nn.Parameter(
            torch.randn(self.num_k, self.num_orb, self.num_nuclei) / (self.num_nuclei ** 0.5)
        )

    def forward(
        self,
        dist_en: torch.Tensor,
    ) -> torch.Tensor:
        if dist_en.dim() == 4:
            dist_en = dist_en.squeeze(-1)
            
        alpha = torch.exp(self.log_alpha)
        B, N, M = dist_en.shape
        K, O, _ = alpha.shape
        
        dist_expanded = dist_en.unsqueeze(2).unsqueeze(3)
        alpha_expanded = alpha.view(1, 1, K, O, M)
        weights_expanded = self.weights.view(1, 1, K, O, M)
        
        decay = torch.exp(-alpha_expanded * dist_expanded)
        envelope = torch.sum(weights_expanded * decay, dim=-1)
        
        envelope = envelope.permute(0, 2, 1, 3).contiguous()
        return envelope


# --- Prototype 4: KineticEnergyCalculator ---
class KineticMode(str, Enum):
    REVERSE = "reverse"
    FORWARD_JVP = "forward_jvp"
    HUTCHINSON = "hutchinson"

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
                    laplacian = laplacian + torch.autograd.grad(jvp_grad.sum(), r_eval, retain_graph=True)[0].reshape(B, dim_total)[:, d]
                    
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


# --- Prototype 5: PsiformerAttentionBlock ---
class PsiformerAttentionBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        mlp_ratio: int = 2,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = 1.0 / (self.head_dim ** 0.5)
        
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
    ) -> torch.Tensor:
        B, N, C = h.shape
        
        norm_h = self.norm1(h)
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


# ============================================================================
# Empirical Test Cases
# ============================================================================

class TestPrototypeEmpirical(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.batch_size = 4
        self.nuclear_coords = torch.tensor([[0.0, 0.0, -0.7], [0.0, 0.0, 0.7]], dtype=torch.float32)

    # ------------------------------------------------------------------------
    # 1. GeometricContext Tests
    # ------------------------------------------------------------------------
    def test_geometric_context_shapes_and_values(self):
        """Test GeometricContext shape and mathematical consistency."""
        r = torch.randn(self.batch_size, 4, 3)
        ctx = GeometricContext.compute(r, self.nuclear_coords)
        
        self.assertEqual(ctx.r_en.shape, (self.batch_size, 4, 2, 3))
        self.assertEqual(ctx.d_en.shape, (self.batch_size, 4, 2, 1))
        self.assertEqual(ctx.r_ee.shape, (self.batch_size, 4, 4, 3))
        self.assertEqual(ctx.d_ee.shape, (self.batch_size, 4, 4, 1))
        
        # Test N=1 edge case
        r_single = torch.randn(self.batch_size, 1, 3)
        ctx_single = GeometricContext.compute(r_single, self.nuclear_coords)
        self.assertIsNone(ctx_single.r_ee)
        self.assertIsNone(ctx_single.d_ee)
        self.assertEqual(ctx_single.r_en.shape, (self.batch_size, 1, 2, 3))

    def test_geometric_context_device_and_dtype_mismatch(self):
        """Challenge: Check what happens if nuclear_coords device differs from r."""
        # Type promotion happens on CPU, but device mismatch or non-tensor would fail.
        # Check if non-tensor raises AttributeError / TypeError
        with self.assertRaises(AttributeError):
            _ = GeometricContext.compute("invalid_tensor", self.nuclear_coords)

    # ------------------------------------------------------------------------
    # 2. FermiInteractionLayerModern Tests
    # ------------------------------------------------------------------------
    def test_fermi_interaction_layer_h2_molecule_bug(self):
        """
        CRITICAL CHALLENGE: For N_up=1, N_down=1 (e.g. H2 molecule),
        mean_h2_up is zeroed out by:
            mean_h2_up = sum_h2_up / self.denom_up if self.n_up > 1 else torch.zeros_like(sum_h2_up)
        Check if opposite-spin electron interaction is erroneously zeroed!
        """
        n_up = 1
        n_down = 1
        d_single = 8
        d_double = 4
        
        layer_modern = FermiInteractionLayerModern(
            d_in_single=d_single,
            d_in_double=d_double,
            d_out_single=d_single,
            d_out_double=d_double,
            n_up=n_up,
            n_down=n_down,
        )
        
        h1 = torch.randn(self.batch_size, 2, d_single)
        # Construct non-zero h2 features between electron 0 (up) and electron 1 (down)
        h2 = torch.ones(self.batch_size, 2, 2, d_double)
        
        # In original layers.py:
        layer_orig = FermiInteractionLayer(
            in_dim_one=d_single,
            in_dim_two=d_double,
            out_dim_one=d_single,
            out_dim_two=d_double,
        )
        
        # In original layers.py, for electron 1 (down), sum_h2_up[:, 1, :] is h2[:, 1, 0, :] = 1.0
        # counts_up for electron 1 is float(n_up) = 1.0.
        # So mean_h2_up[:, 1, :] is 1.0!
        # But in FermiInteractionLayerModern:
        # self.n_up > 1 is False -> mean_h2_up is ALL ZEROS!
        # Let's verify this empirically:
        # Trace mean_h2_up inside modern layer:
        sum_h2_up = torch.sum(h2[:, :, :n_up, :], dim=2)
        diag_up = torch.arange(n_up, device=h2.device)
        sum_h2_up[:, diag_up, :] = sum_h2_up[:, diag_up, :] - h2[:, diag_up, diag_up, :]
        mean_h2_up_modern = sum_h2_up / layer_modern.denom_up if n_up > 1 else torch.zeros_like(sum_h2_up)
        
        print(f"\n[Empirical Check] n_up=1, n_down=1:")
        print(f"Modern mean_h2_up max: {mean_h2_up_modern.abs().max().item()}")
        
        # In original layer:
        eye_mask = torch.eye(2, dtype=torch.bool)
        h2_no_diag = h2.masked_fill(eye_mask.unsqueeze(0).unsqueeze(-1), 0.0)
        sum_h2_up_orig = torch.sum(h2_no_diag[:, :, :n_up, :], dim=2)
        counts_up = torch.full((1, 2, 1), float(n_up))
        counts_up[:, :n_up, :] = float(n_up - 1)
        mean_h2_up_orig = torch.where(counts_up > 0, sum_h2_up_orig / counts_up.clamp(min=1.0), torch.zeros_like(sum_h2_up_orig))
        print(f"Original mean_h2_up max: {mean_h2_up_orig.abs().max().item()}")
        print(f"Original mean_h2_up for electron 1 (down seeing up): {mean_h2_up_orig[0, 1, 0].item()}")
        
        # PROOF OF BUG:
        self.assertEqual(mean_h2_up_modern.abs().max().item(), 0.0)
        self.assertEqual(mean_h2_up_orig[0, 1, 0].item(), 1.0)
        print(">>> CONFIRMED BUG in Prototype 2: n_up=1 erroneously zeroes opposite-spin mean_h2_up!")

    def test_fermi_interaction_layer_autograd_second_derivative(self):
        """Test if in-place slice subtraction in Prototype 2 breaks double backward."""
        n_up = 2
        n_down = 2
        d_single = 8
        d_double = 4
        layer = FermiInteractionLayerModern(d_single, d_double, d_single, d_double, n_up, n_down)
        
        h1 = torch.randn(self.batch_size, 4, d_single, requires_grad=True)
        h2 = torch.randn(self.batch_size, 4, 4, d_double, requires_grad=True)
        
        h1_out, h2_out = layer(h1, h2)
        loss = h1_out.sum()
        
        # 1st backward
        g = torch.autograd.grad(loss, h2, create_graph=True)[0]
        # 2nd backward (kinetic energy requires second derivative)
        g2 = torch.autograd.grad(g.sum(), h2)[0]
        self.assertTrue(torch.isfinite(g2).all())

    # ------------------------------------------------------------------------
    # 3. NuclearEnvelopeModern Tests
    # ------------------------------------------------------------------------
    def test_nuclear_envelope_shapes_and_5d_broadcast(self):
        """Test NuclearEnvelopeModern shapes and check if 5D tensor still exists."""
        K = 2
        N_orb = 3
        N = 4
        M = 2
        env = NuclearEnvelopeModern(num_determinants=K, num_orbitals=N_orb, nuclear_coords=self.nuclear_coords)
        dist_en = torch.rand(self.batch_size, N, M)
        
        out = env(dist_en)
        self.assertEqual(out.shape, (self.batch_size, K, N, N_orb))
        
        # Check parameter compatibility with original NuclearEnvelope
        orig_env = NuclearEnvelope(num_determinants=K, num_orbitals=N_orb, num_nuclei=M)
        self.assertIn("decay_rates_raw", orig_env.state_dict())
        self.assertNotIn("decay_rates_raw", env.state_dict())
        self.assertIn("log_alpha", env.state_dict())
        print(f"\n[Empirical Check] NuclearEnvelope parameter mismatch: "
              f"Original uses 'decay_rates_raw' (softplus), Modern uses 'log_alpha' (exp)")

    # ------------------------------------------------------------------------
    # 4. KineticEnergyCalculator Tests
    # ------------------------------------------------------------------------
    def test_kinetic_calculator_all_modes_numeric(self):
        """
        Compare REVERSE, FORWARD_JVP, and HUTCHINSON on a known analytical function.
        f(r) = sum(r_i^2)
        grad = 2 * r
        laplacian = 2 * dim_total = 2 * (N * 3)
        grad_squared = 4 * sum(r^2)
        """
        B = 2
        N = 2
        dim_total = N * 3
        
        def mock_log_abs_psi(r):
            # log|psi| = sum(r^2)
            return torch.sum(r ** 2, dim=(-1, -2))
        
        r = torch.randn(B, N, 3, dtype=torch.float64)
        
        # 1. Reverse mode
        t_rev, lap_rev, g2_rev = KineticEnergyCalculator.compute_kinetic_energy(
            mock_log_abs_psi, r, mode=KineticMode.REVERSE
        )
        
        # Analytical expected laplacian: 2 * dim_total = 2 * 6 = 12.0
        expected_lap = torch.full((B,), 2.0 * dim_total, dtype=torch.float64)
        self.assertTrue(torch.allclose(lap_rev, expected_lap, atol=1e-10))
        
        # 2. Forward JVP mode
        t_jvp, lap_jvp, g2_jvp = KineticEnergyCalculator.compute_kinetic_energy(
            mock_log_abs_psi, r, mode=KineticMode.FORWARD_JVP
        )
        self.assertTrue(torch.allclose(lap_jvp, expected_lap, atol=1e-10))
        self.assertTrue(torch.allclose(t_rev, t_jvp, atol=1e-10))
        
        # 3. Hutchinson mode (average over 500 probes for law of large numbers)
        t_hutch, lap_hutch, g2_hutch = KineticEnergyCalculator.compute_kinetic_energy(
            mock_log_abs_psi, r, mode=KineticMode.HUTCHINSON, num_probes=500
        )
        print(f"\n[Empirical Check] Kinetic Energy Laplacian:")
        print(f"Analytical: {expected_lap[0].item():.4f}")
        print(f"Reverse:    {lap_rev[0].item():.4f}")
        print(f"Forward JVP:{lap_jvp[0].item():.4f}")
        print(f"Hutchinson (500 probes): {lap_hutch[0].item():.4f}")
        self.assertTrue(torch.allclose(lap_hutch, expected_lap, rtol=0.10))

    def test_kinetic_calculator_forward_jvp_speed_and_overhead(self):
        """
        CHALLENGE: Measure wall-clock time and forward-pass counts of FORWARD_JVP vs REVERSE.
        Verify if FORWARD_JVP is actually 3x faster or if it executes (3N+1) forward passes!
        """
        B = 16
        N = 6  # 18 dims
        dim_total = N * 3
        
        forward_call_count = {"count": 0}
        def dummy_nn_forward(r):
            forward_call_count["count"] += 1
            # simulate a small non-linear computation
            return torch.sum(torch.sin(r), dim=(-1, -2))
        
        r = torch.randn(B, N, 3)
        
        # Test REVERSE
        forward_call_count["count"] = 0
        t0 = time.perf_counter()
        KineticEnergyCalculator.compute_kinetic_energy(dummy_nn_forward, r, mode=KineticMode.REVERSE)
        t_rev = time.perf_counter() - t0
        calls_rev = forward_call_count["count"]
        
        # Test FORWARD_JVP
        forward_call_count["count"] = 0
        t0 = time.perf_counter()
        KineticEnergyCalculator.compute_kinetic_energy(dummy_nn_forward, r, mode=KineticMode.FORWARD_JVP)
        t_jvp = time.perf_counter() - t0
        calls_jvp = forward_call_count["count"]
        
        print(f"\n[Empirical Check] Kinetic Energy Mode Comparison (N={N}, dim_total={dim_total}):")
        print(f"Reverse Mode:     Time = {t_rev*1000:.2f} ms, Forward Calls = {calls_rev}")
        print(f"Forward JVP Mode: Time = {t_jvp*1000:.2f} ms, Forward Calls = {calls_jvp}")
        
        # PROOF: FORWARD_JVP performs 1 + dim_total forward calls, whereas REVERSE performs only 1!
        self.assertEqual(calls_rev, 1)
        self.assertEqual(calls_jvp, 1 + dim_total)
        self.assertGreater(t_jvp, t_rev)
        print(">>> CONFIRMED: Forward JVP in Prototype 4 is SLOWER than Reverse because of 3N additional forward calls!")

    # ------------------------------------------------------------------------
    # 5. PsiformerAttentionBlock Tests
    # ------------------------------------------------------------------------
    def test_psiformer_attention_equivariance_and_n1(self):
        """Test Psiformer equivariance under electron permutation, and N=1 behavior."""
        embed_dim = 16
        num_heads = 4
        block = PsiformerAttentionBlock(embed_dim=embed_dim, num_heads=num_heads)
        
        # 1. Test N=1
        h_single = torch.randn(self.batch_size, 1, embed_dim)
        out_single = block(h_single)
        self.assertEqual(out_single.shape, (self.batch_size, 1, embed_dim))
        
        # 2. Test Permutation Equivariance for N=3
        N = 3
        h = torch.randn(self.batch_size, N, embed_dim)
        r = torch.randn(self.batch_size, N, 3)
        r_ee = r.unsqueeze(2) - r.unsqueeze(1)
        d_ee = torch.norm(r_ee, dim=-1, keepdim=True)
        
        out1 = block(h, r_ee, d_ee)
        
        # Permute electrons: perm = [1, 2, 0]
        perm = [1, 2, 0]
        h_perm = h[:, perm, :]
        r_ee_perm = r_ee[:, perm, :][:, :, perm]
        d_ee_perm = d_ee[:, perm, :][:, :, perm]
        
        out_perm = block(h_perm, r_ee_perm, d_ee_perm)
        # Check if out_perm == out1[:, perm, :]
        diff = torch.max(torch.abs(out_perm - out1[:, perm, :])).item()
        print(f"\n[Empirical Check] Psiformer Permutation Equivariance Max Diff: {diff:.2e}")
        self.assertLess(diff, 1e-6)

    def test_nuclear_envelope_still_allocates_5d_tensor(self):
        """
        CRITICAL CHALLENGE: Check if NuclearEnvelopeModern truly eliminated the 5D broadcast tensor
        or if it still creates a (B, N, K, O, M) 5D tensor.
        """
        B = 4
        N = 6
        M = 2
        K = 4
        O = 4
        
        env = NuclearEnvelopeModern(num_determinants=K, num_orbitals=O, nuclear_coords=self.nuclear_coords)
        dist_en = torch.randn(B, N, M)
        
        alpha = torch.exp(env.log_alpha)
        dist_expanded = dist_en.unsqueeze(2).unsqueeze(3)
        alpha_expanded = alpha.view(1, 1, K, O, M)
        
        # This is line 768 of the proposal:
        decay = torch.exp(-alpha_expanded * dist_expanded)
        
        print(f"\n[Empirical Check] NuclearEnvelopeModern 'decay' tensor dimensions:")
        print(f"decay.shape: {decay.shape} (ndim={decay.ndim})")
        print(f"Total elements: {decay.numel()}")
        
        # PROOF: decay is STILL a 5D tensor!
        self.assertEqual(decay.ndim, 5)
        self.assertEqual(decay.shape, (B, N, K, O, M))
        print(">>> CONFIRMED: NuclearEnvelopeModern STILL constructs a 5D tensor! Einsum claim is not implemented.")

    def test_fermi_interaction_layer_fix_verification(self):
        """
        Empirically verify the correct vectorized fix for FermiInteractionLayerModern:
        Use boolean mask / counts per electron instead of scalar `if self.n_up > 1`.
        """
        n_up = 1
        n_down = 1
        B = 2
        d_double = 4
        
        # h2 is non-zero between electron 0 (up) and electron 1 (down)
        h2 = torch.ones(B, 2, 2, d_double)
        
        # Correct counts tensor:
        counts_up = torch.full((1, 2, 1), float(n_up))
        counts_up[:, :n_up, :] = float(n_up - 1)  # 0 for up electrons, 1 for down electrons
        
        sum_h2_up = torch.sum(h2[:, :, :n_up, :], dim=2)
        diag_up = torch.arange(n_up)
        sum_h2_up[:, diag_up, :] = sum_h2_up[:, diag_up, :] - h2[:, diag_up, diag_up, :]
        
        # Correct pooling:
        mean_h2_up_fixed = torch.where(counts_up > 0, sum_h2_up / counts_up.clamp(min=1.0), torch.zeros_like(sum_h2_up))
        
        # Electron 0 (up) has no other up electrons: mean_h2_up is 0
        self.assertEqual(mean_h2_up_fixed[0, 0, 0].item(), 0.0)
        # Electron 1 (down) sees electron 0: mean_h2_up is 1.0!
        self.assertEqual(mean_h2_up_fixed[0, 1, 0].item(), 1.0)
        print("\n[Empirical Check] Proposed fix for FermiInteractionLayer successfully preserves opposite-spin correlation!")


if __name__ == "__main__":
    unittest.main()

