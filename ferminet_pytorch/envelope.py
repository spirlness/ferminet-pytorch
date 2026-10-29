"""Nuclear envelopes evaluated from precomputed electron–nucleus distances."""

from typing import Optional
import math

import torch
from torch import nn
from torch.nn import functional as F


class NuclearEnvelope(nn.Module):
    """Sum weighted nuclear decays into orbital envelopes.

    Input distances have shape ``(B, N, M)``; output has shape ``(B, K, N, O)``.
    Geometry and distance smoothing belong to the caller. Both model families
    use this implementation, including its parameter initialization.
    """

    def __init__(
        self,
        num_determinants: int,
        num_orbitals: int,
        num_nuclei: int,
        init_decay_rate: float = 1.0,
        envelope_type: str = "exponential",
        nuclear_charges: Optional[torch.Tensor] = None,
        envelope_scale: float = 2.0,
    ) -> None:
        super().__init__()
        if min(num_determinants, num_orbitals, num_nuclei) < 1:
            raise ValueError("Envelope dimensions must be positive")
        if envelope_type not in ("exponential", "gaussian", "soft_exponential"):
            raise ValueError("Unknown envelope_type")
        if not math.isfinite(envelope_scale) or envelope_scale <= 0:
            raise ValueError("envelope_scale must be positive and finite")
        if init_decay_rate <= 0:
            raise ValueError("init_decay_rate must be positive")
        if nuclear_charges is not None and nuclear_charges.shape != (num_nuclei,):
            raise ValueError("nuclear_charges must have shape (num_nuclei,)")
        self.num_nuclei = num_nuclei
        self.envelope_type = envelope_type
        self.envelope_scale = float(envelope_scale)
        shape = (num_determinants, num_orbitals, num_nuclei)
        self.weights = nn.Parameter(torch.empty(shape))
        self.decay_rates_raw = nn.Parameter(torch.empty(shape))
        nn.init.normal_(self.weights, mean=1.0, std=0.1)
        rates = (
            torch.full((num_nuclei,), init_decay_rate)
            if nuclear_charges is None
            else nuclear_charges.detach()
        )
        rates = rates.to(self.decay_rates_raw).clamp_min(1e-4)
        raw = rates + torch.log(-torch.expm1(-rates))
        with torch.no_grad():
            self.decay_rates_raw.copy_(raw.expand_as(self.decay_rates_raw))

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        if distances.ndim != 3 or distances.shape[-1] != self.num_nuclei:
            raise ValueError("distances must have shape (batch, electrons, nuclei)")
        radial = distances.square() if self.envelope_type == "gaussian" else distances
        if self.envelope_type == "soft_exponential":
            radial = (
                2
                * distances.square()
                / (torch.sqrt(1 + (distances / self.envelope_scale).square()) + 1)
            )
        alpha = F.softplus(self.decay_rates_raw)
        decay = torch.exp(-radial[:, None, :, None, :] * alpha[None, :, None, :, :])
        return (self.weights[None, :, None, :, :] * decay).sum(dim=-1)
