"""Correlation-aware statistics for parallel MCMC walker chains."""

import math
import warnings
from typing import Dict, Tuple, Union

import torch


MCMCStatistic = Union[float, int, str]

# 低于该时间步数时，"未检测到正自相关" 不是一个有意义的结论。
# 对短链扣除时间均值会系统性地引入约 1/T 的负自相关偏置，使 Geyer 初始正序列
# 在首个 lag 即截断。Sokal 判据要求链长 N >= 50*tau，对 tau ~ 1 即 N >= 50，
# 故取该量级作为"能否判定确实无自相关"的下界。
MIN_TIME_FOR_AUTOCORRELATION = 50

# 链长少于该倍数个自相关时间时，tau 的有限样本偏差已不可忽略，发出警告。
MIN_TIME_PER_TAU = 20.0


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


def _statistical_inefficiency(chains: torch.Tensor) -> Tuple[float, str]:
    """Estimate ``g = 1 + 2 sum(rho_t)`` via Geyer's initial positive sequence.

    Returns:
        Tuple[float, str]: ``(inefficiency, status)`` where status is one of

        - ``"ok"``: at least one positive pair was found, so ``g`` is a real estimate;
        - ``"insufficient_samples"``: too few time steps to form a single pair;
        - ``"unresolved"``: the sequence truncated at the very first pair **and** the
          chain is too short for "no positive autocorrelation" to be a meaningful
          finding. Returning ``g = 1`` here would be indistinguishable from a genuine
          uncorrelated chain and would silently report full efficiency, so the value
          is reported as ``nan`` instead.
    """
    rho = _mean_autocorrelation(chains)
    time = int(chains.shape[0])
    if rho.numel() < 3:
        return float("nan"), "insufficient_samples"

    paired_sums = []
    for lag in range(1, rho.numel() - 1, 2):
        pair = float(rho[lag] + rho[lag + 1])
        if not math.isfinite(pair) or pair <= 0.0:
            break
        paired_sums.append(pair)

    if not paired_sums:
        # Truncated at the first pair. For a long chain this is a legitimate
        # "no detectable autocorrelation" result; for a short one it is the
        # mean-subtraction artifact and carries no information.
        if time < MIN_TIME_FOR_AUTOCORRELATION:
            return float("nan"), "unresolved"
        return 1.0, "ok"

    # Initial monotone sequence prevents noisy late pairs from increasing again.
    for index in range(1, len(paired_sums)):
        paired_sums[index] = min(paired_sums[index], paired_sums[index - 1])
    return max(1.0, 1.0 + 2.0 * sum(paired_sums)), "ok"


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

    When the autocorrelation time cannot be resolved from the supplied series, the
    derived quantities ``tau_int``, ``statistical_inefficiency``, ``ess`` and
    ``energy_sem`` are reported as ``nan`` and a ``UserWarning`` is emitted, rather
    than silently reporting full statistical efficiency. Inspect
    ``autocorrelation_status`` for the reason.

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
    time = int(chains.shape[0])
    total = chains.numel()
    mean = float(chains.mean())
    variance = float(chains.var(unbiased=True))

    if variance == 0.0:
        # A constant series carries no uncertainty; ESS = N and SEM = 0 are exact.
        inefficiency = 1.0
        ess = float(total)
        sem = 0.0
        autocorrelation_status = "constant"
    else:
        inefficiency, autocorrelation_status = _statistical_inefficiency(chains)
        if autocorrelation_status == "ok":
            ess = min(float(total), max(1.0, float(total) / inefficiency))
            sem = math.sqrt(variance / ess)
            tau_int = 0.5 * inefficiency
            if time < MIN_TIME_PER_TAU * tau_int:
                warnings.warn(
                    f"MCMC series has only {time} time steps but tau_int = {tau_int:.2f}; "
                    f"the autocorrelation time is under-resolved (need about "
                    f"{MIN_TIME_PER_TAU * tau_int:.0f} steps), so tau_int, ESS and the "
                    f"standard error carry a substantial finite-sample bias.",
                    UserWarning,
                    stacklevel=2,
                )
        else:
            inefficiency = float("nan")
            ess = float("nan")
            sem = float("nan")
            if autocorrelation_status == "unresolved":
                reason = (
                    f"the autocorrelation sequence truncated at the first lag and the "
                    f"chain has only {time} time steps (need at least "
                    f"{MIN_TIME_FOR_AUTOCORRELATION}); mean subtraction on a short chain "
                    f"induces a spurious negative autocorrelation, so no correlation "
                    f"structure can be resolved"
                )
            else:
                reason = (
                    f"the chain has only {time} time steps, too few to form a single lag pair"
                )
            warnings.warn(
                f"Cannot resolve the MCMC autocorrelation time: {reason}. "
                f"Reporting tau_int, ESS and energy_sem as nan instead of assuming "
                f"independent samples. Increase the number of time steps "
                f"(train_vmc evaluation_steps) and re-run.",
                UserWarning,
                stacklevel=2,
            )

    r_hat, r_hat_status = _split_r_hat(chains)
    return {
        "energy_mean": mean,
        "energy_variance": variance,
        "energy_sem": sem,
        "ess": ess,
        "tau_int": 0.5 * inefficiency,
        "statistical_inefficiency": inefficiency,
        "autocorrelation_status": autocorrelation_status,
        "r_hat": r_hat,
        "r_hat_status": r_hat_status,
        "num_steps": time,
        "num_walkers": int(chains.shape[1]),
        "num_samples": int(total),
    }
