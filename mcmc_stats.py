"""Correlation-aware statistics for parallel MCMC walker chains."""

import math
from typing import Dict, Union

import torch


MCMCStatistic = Union[float, int, str]


def _mean_autocorrelation(chains: torch.Tensor) -> torch.Tensor:
    """Estimate the pooled normalized autocorrelation using FFT autocovariances."""
    time = chains.shape[0]
    centered = chains - chains.mean(dim=0, keepdim=True)
    fft_size = 1 << (2 * time - 1).bit_length()
    spectrum = torch.fft.rfft(centered, n=fft_size, dim=0)
    autocov = torch.fft.irfft(spectrum * spectrum.conj(), n=fft_size, dim=0)[:time]
    denominators = torch.arange(time, 0, -1, dtype=chains.dtype).unsqueeze(1)
    autocov = autocov / denominators
    pooled = autocov.mean(dim=1)
    if pooled[0] <= 0 or not torch.isfinite(pooled[0]):
        return torch.ones(1, dtype=chains.dtype)
    return pooled / pooled[0]


def _statistical_inefficiency(chains: torch.Tensor) -> float:
    """Return ``1 + 2 sum(rho_t)`` using Geyer's initial positive sequence."""
    rho = _mean_autocorrelation(chains)
    if rho.numel() < 3:
        return 1.0
    paired_sums = []
    for lag in range(1, rho.numel() - 1, 2):
        pair = float(rho[lag] + rho[lag + 1])
        if not math.isfinite(pair) or pair <= 0.0:
            break
        paired_sums.append(pair)
    # Initial monotone sequence prevents noisy late pairs from increasing again.
    for index in range(1, len(paired_sums)):
        paired_sums[index] = min(paired_sums[index], paired_sums[index - 1])
    return max(1.0, 1.0 + 2.0 * sum(paired_sums))


def _split_r_hat(chains: torch.Tensor) -> tuple[float, str]:
    """Compute classic split-R-hat across walker chains."""
    time, walkers = chains.shape
    half = time // 2
    if half < 2 or walkers < 2:
        return float("nan"), "insufficient_samples"
    split = torch.cat((chains[:half], chains[-half:]), dim=1)
    chain_means = split.mean(dim=0)
    within = split.var(dim=0, unbiased=True).mean()
    between = half * chain_means.var(unbiased=True)
    if within == 0:
        if between == 0:
            return 1.0, "constant_chains"
        return float("inf"), "zero_within_chain_variance"
    variance_hat = ((half - 1.0) / half) * within + between / half
    return float(torch.sqrt(variance_hat / within)), "ok"


def estimate_mcmc_statistics(energies: torch.Tensor) -> Dict[str, MCMCStatistic]:
    """Estimate mean, variance, autocorrelation time, ESS, SEM, and split-R-hat.

    Args:
        energies: Local-energy samples with shape ``(time, walkers)``.
    """
    if energies.ndim != 2:
        raise ValueError(f"energies must have shape (time, walkers), got {tuple(energies.shape)}")
    if energies.shape[0] < 2 or energies.shape[1] < 1:
        raise ValueError("at least two time samples and one walker are required")
    if not torch.isfinite(energies).all():
        count = int((~torch.isfinite(energies)).sum())
        raise FloatingPointError(f"MCMC energy series contains {count} non-finite values")

    chains = energies.detach().to(device="cpu", dtype=torch.float64)
    total = chains.numel()
    mean = float(chains.mean())
    variance = float(chains.var(unbiased=True))
    if variance == 0.0:
        inefficiency = 1.0
        ess = float(total)
        sem = 0.0
    else:
        inefficiency = _statistical_inefficiency(chains)
        ess = min(float(total), max(1.0, float(total) / inefficiency))
        sem = math.sqrt(variance / ess)
    r_hat, r_hat_status = _split_r_hat(chains)
    return {
        "energy_mean": mean,
        "energy_variance": variance,
        "energy_sem": sem,
        "ess": ess,
        "tau_int": 0.5 * inefficiency,
        "statistical_inefficiency": inefficiency,
        "r_hat": r_hat,
        "r_hat_status": r_hat_status,
        "num_steps": int(chains.shape[0]),
        "num_walkers": int(chains.shape[1]),
        "num_samples": int(total),
    }
