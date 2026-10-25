"""Independent, frozen-wavefunction evaluation with traceable uncertainty."""

import os
import time
from typing import Any, Callable, Dict, Optional, Union

import torch
from torch import nn

from .hamiltonian import MolecularSystem, compute_local_energy
from .mcmc import MCMCSampler
from .vmc_statistics import energy_statistics, apply_reference
from .evaluation_observations import ObservationWriter


def evaluate_frozen_wavefunction(
    model: nn.Module,
    system: MolecularSystem,
    num_walkers: int = 1024,
    num_steps: int = 100,
    burn_in_steps: int = 40,
    decorrelation_steps: int = 10,
    step_size: float = 0.5,
    target_acceptance: Optional[float] = None,
    move_mode: str = "all",
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = torch.float32,
    seed: int = 2024,
    verbose: bool = True,
    exact_energy: Optional[float] = None,
    hf_energy: Optional[float] = None,
    reference_label: Optional[str] = None,
    reference_source: Optional[str] = None,
    max_seconds: Optional[float] = None,
    trajectory_path: Optional[str] = None,
    energy_batch_size: Optional[int] = None,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
    observations_path: Optional[str] = None,
    records_per_shard: int = 128,
) -> Dict[str, Any]:
    """Approximate intervals conditional on this checkpoint, not training error.

    max_seconds bounds production sampling independently of observed energies.
    Every observation must be finite; invalid samples are never discarded.
    """
    if num_steps < 1 or num_walkers < 1 or decorrelation_steps < 1:
        raise ValueError(
            "Evaluation requires positive steps, walkers and decorrelation"
        )
    if energy_batch_size is not None and energy_batch_size < 1:
        raise ValueError("Energy batch size must be positive")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    target_device = (
        torch.device(device) if device is not None else system.nuclear_coords.device
    )
    start = time.perf_counter()
    trajectory = []
    writer = (
        ObservationWriter(observations_path, records_per_shard, num_walkers, seed)
        if observations_path is not None
        else None
    )
    observations_complete = False
    was_training = model.training
    model.eval()
    try:
        sampler = MCMCSampler(
            model,
            system,
            num_walkers=num_walkers,
            step_size=step_size,
            target_acceptance=target_acceptance,
            move_mode=move_mode,
            device=target_device,
            dtype=dtype,
        )
        if progress_callback is None:
            sampler.burn_in(burn_in_steps)
        else:
            progress_callback(
                dict(
                    status="evaluation_burn_in",
                    burn_in_steps=0,
                    requested_burn_in_steps=burn_in_steps,
                )
            )
            for step in range(0, burn_in_steps, 25):
                if should_stop is not None and should_stop():
                    raise InterruptedError("Frozen evaluation interrupted")
                sampler.burn_in(min(25, burn_in_steps - step))
                progress_callback(
                    dict(
                        status="evaluation_burn_in",
                        burn_in_steps=min(step + 25, burn_in_steps),
                        requested_burn_in_steps=burn_in_steps,
                    )
                )
        production_start = time.perf_counter()
        for _ in range(num_steps):
            if should_stop is not None and should_stop():
                raise InterruptedError("Frozen evaluation interrupted")
            if (
                trajectory
                and max_seconds is not None
                and time.perf_counter() - production_start >= max_seconds
            ):
                break
            walkers = sampler.sample(decorrelation_steps, adapt=False)
            batches = walkers.split(energy_batch_size or num_walkers)
            chunks = []
            for batch in batches:
                r = batch.detach().clone().requires_grad_(True)
                if writer is not None:
                    energy, kinetic, potential, log_abs, sign = compute_local_energy(
                        model, system, r, return_wavefunction=True
                    )
                    # Concatenate on device before one host transfer per record.
                    # No arithmetic or RNG order changes: each value is copied
                    # from the same eager, exact-energy calculation as before.
                    chunks.append(
                        torch.stack(
                            (
                                energy.detach(),
                                kinetic.detach(),
                                potential.detach(),
                                log_abs,
                                sign,
                            )
                        )
                    )
                else:
                    energy, _, _ = compute_local_energy(model, system, r)
                    chunks.append(energy.detach())
            record = torch.cat(chunks, dim=1 if writer is not None else 0).cpu()
            energies = record[0] if writer is not None else record
            if not torch.isfinite(energies).all():
                raise FloatingPointError(
                    "Nonfinite energy in frozen evaluation; no observations were discarded"
                )
            if writer is not None:
                writer.append(
                    positions=walkers.detach().cpu(),
                    energy=energies,
                    kinetic=record[1],
                    potential=record[2],
                    log_abs=record[3],
                    sign=record[4],
                )
            trajectory.append(energies.detach().cpu())
            if progress_callback is not None:
                progress_callback(
                    dict(
                        status="evaluating",
                        time_records=len(trajectory),
                        requested_time_records=num_steps,
                        evaluation_seconds=time.perf_counter() - production_start,
                        force=len(trajectory) in (1, num_steps),
                    )
                )
        series = torch.stack(trajectory)
        result = energy_statistics(series)
        result.update(
            decorrelation_steps=decorrelation_steps,
            eval_time_sec=time.perf_counter() - start,
            time_records=len(trajectory),
            requested_time_records=num_steps,
            energy_batch_size=energy_batch_size or num_walkers,
            burn_in_seconds=production_start - start,
            production_seconds=time.perf_counter() - production_start,
            seed=seed,
            stopping_reason="completed"
            if len(trajectory) == num_steps
            else "time_budget",
            mcmc_acceptance=sampler.last_acceptance_rate,
            mcmc_step_size=sampler.step_size,
            spin_projection=getattr(
                getattr(model, "spin_projector", None), "mode", "none"
            ),
        )
        if trajectory_path is not None:
            torch.save(series, trajectory_path)
            result["trajectory_path"] = os.path.abspath(trajectory_path)
        observations_complete = len(trajectory) == num_steps
    finally:
        try:
            if writer is not None:
                observation_manifest = writer.close(complete=observations_complete)
        finally:
            model.train(was_training)
    if writer is not None:
        result["observations_manifest"] = observation_manifest

    # Reference matching includes electron number, geometry and target spin.
    projector = getattr(model, "spin_projector", None)
    charges = system.nuclear_charges.detach().cpu().double()
    nuclei = system.nuclear_coords.detach().cpu().double()
    if (
        exact_energy is None
        and system.n_up == system.n_down == 1
        and getattr(projector, "mode", None) == "singlet"
    ):
        if charges.shape == (2,) and torch.equal(
            charges, torch.ones(2, dtype=torch.float64)
        ):
            bond = float(torch.linalg.vector_norm(nuclei[0] - nuclei[1]))
            if abs(bond - 1.4) <= 1e-6:
                exact_energy = -1.174475714220
                hf_energy = -1.13362957 if hf_energy is None else hf_energy
                reference_label = reference_label or "H2 BO singlet, R=1.4 bohr"
                reference_source = (
                    reference_source or "https://doi.org/10.1103/PhysRevA.82.032509"
                )
    apply_reference(result, exact_energy, hf_energy, reference_label, reference_source)
    if verbose:
        print(
            f"Frozen energy: {result['energy_mean']:.8f} +/- {result['energy_sem']:.8f} Ha; "
            f"diagnostics={result['diagnostics_status']}; chemical_accuracy={result['chemical_accuracy_status']}"
        )
    return result
