"""Observable-specific MCMC diagnostics; approximate intervals, not certificates.

Geyer initial positive/monotone sequence estimates the sum of autocorrelations.
Reblocking preserves the time axis; independent walkers are never flattened
into a fictitious time series. Diagnostics do not prove global ergodicity.
"""
import math

import torch


def autocorrelation_time(series):
    x = torch.as_tensor(series).detach().to(device='cpu', dtype=torch.float64)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2 or x.numel() == 0:
        raise ValueError('Expected a nonempty (time, walkers) series')
    if not torch.isfinite(x).all():
        raise FloatingPointError('MCMC diagnostics require all finite observations')
    length = x.shape[0]
    if length < 2:
        return 0., 1., 1.
    centered = x - x.mean()
    variance = centered.square().mean()
    if variance == 0:
        return 0., 1., 1.
    n_fft = 1 << (2 * length - 1).bit_length()
    spectrum = torch.fft.rfft(centered, n=n_fft, dim=0)
    cov = torch.fft.irfft(spectrum * spectrum.conj(), n=n_fft, dim=0)[:length].mean(1)
    cov = cov / torch.arange(length, 0, -1, dtype=torch.float64)
    rho = cov / cov[0]
    pairs = []
    previous = float('inf')
    for lag in range(0, length - 1, 2):
        pair = float(rho[lag] + rho[lag+1])
        if pair <= 0:
            break
        previous = min(previous, pair)
        pairs.append(previous)
    tau = min(float(length), max(1., -1. + 2. * sum(pairs)))
    return float(rho[1].clamp(-.999, .999)), tau, 1. / tau


def split_rhat(series):
    x = series.double().cpu()
    half = x.shape[0] // 2
    if half < 2:
        return float('inf')
    split = torch.cat((x[:half], x[-half:]), dim=1)
    within = split.var(dim=0, unbiased=True).mean()
    between = half * split.mean(0).var(unbiased=True)
    if within == 0:
        return 1. if between == 0 else float('inf')
    variance = (half - 1.) / half * within + between / half
    return max(1., math.sqrt(float(variance / within)))


def energy_statistics(series):
    x = torch.as_tensor(series).detach().to(device='cpu', dtype=torch.float64)
    rho, tau, fraction = autocorrelation_time(x)
    if x.ndim == 1:
        x = x[:, None]
    length, walkers = x.shape
    total = x.numel()
    mean = float(x.mean())
    variance = float(x.var(unbiased=True)) if total > 1 else 0.
    std = math.sqrt(variance)
    ess = max(1., total / tau)
    sem_ess = std / math.sqrt(ess)
    levels = []
    block = 1
    time_means = x.mean(dim=1)
    while length // block >= 16:
        count = length // block
        means = time_means[:count*block].reshape(count, block).mean(1)
        sem = math.sqrt(float(means.var(unbiased=True)) / count)
        levels.append({'block_length': block, 'num_blocks': count, 'sem': sem})
        block *= 2
    plateau = False
    for index in range(2, len(levels)):
        triple = levels[index-2:index+1]
        errors = [v['sem'] for v in triple]
        stable = max(errors) == 0 or (min(errors) > 0 and max(errors)/min(errors) <= 1.25)
        if triple[0]['block_length'] >= 2*tau and stable:
            plateau = True
    sem_block = max([v['sem'] for v in levels] or [sem_ess])
    sem = max(sem_ess, sem_block)
    rhat = split_rhat(x)
    reasons = []
    if length < 128:
        reasons.append('fewer_than_128_time_records')
    if ess < 1000:
        reasons.append('effective_samples_below_1000')
    if rhat > 1.01:
        reasons.append('split_rhat_above_1.01')
    if not plateau:
        reasons.append('no_reblocking_plateau')
    return {
        'total_samples': total, 'num_valid': total, 'num_invalid': 0,
        'energy_mean': mean, 'energy_var': variance, 'energy_std': std,
        'energy_sem': sem, 'energy_sem_naive': std/math.sqrt(total),
        'energy_sem_ess': sem_ess, 'energy_sem_block': sem_block,
        'energy_ci95': [mean-1.96*sem, mean+1.96*sem],
        'interval_method': 'approximate_normal_95_percent',
        'rho_1': rho, 'rho_1_batch': autocorrelation_time(time_means)[0],
        'tau_int': tau, 'ess_fraction': fraction, 'effective_sample_size': ess,
        'split_rhat': rhat, 'reblocking': levels, 'blocking_plateau': plateau,
        'diagnostics_status': 'passed' if not reasons else 'insufficient_evidence',
        'diagnostics_reasons': reasons,
    }


def apply_reference(result, exact_energy=None, hf_energy=None, label=None, source=None):
    result.update(reference_label=label, reference_source=source,
                  chemical_accuracy_met=False, chemical_accuracy_status='no_reference')
    if exact_energy is None:
        return result
    if not math.isfinite(exact_energy):
        raise ValueError('Reference energy must be finite')
    error = abs(result['energy_mean'] - exact_energy)
    result.update(exact_energy=exact_energy, abs_error_ha=error,
                  abs_error_mha=1000*error, abs_error_kcal=627.509474*error)
    if result['diagnostics_status'] != 'passed':
        result['chemical_accuracy_status'] = 'insufficient_evidence'
    else:
        result['chemical_accuracy_met'] = error + 1.96*result['energy_sem'] <= .0015936
        result['chemical_accuracy_status'] = 'met' if result['chemical_accuracy_met'] else 'not_met'
    if hf_energy is not None:
        result['hf_energy'] = hf_energy
        denominator = exact_energy - hf_energy
        result['correlation_retrieved_pct'] = 100*(result['energy_mean']-hf_energy)/denominator if denominator else None
    return result
