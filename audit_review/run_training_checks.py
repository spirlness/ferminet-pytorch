"""Bounded H2/LiH training checks for the scientific review.

This is a convergence smoke test, not a paper reproduction. Results are saved
after every run so an interrupted audit retains completed evidence.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hamiltonian import compute_local_energy
from train_vmc import build_h2_system, build_lih_system, train_vmc


OUTPUT = ROOT / "audit_review" / "training_results.json"


def save(payload: dict) -> None:
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def evaluate_final_model(result: dict, batches: int, steps_between: int) -> dict:
    model = result["model"]
    system = result["system"]
    sampler = result["sampler"]
    sampler.adapt_step_size = False
    model.eval()
    block_means: list[float] = []
    all_energies: list[np.ndarray] = []
    for _ in range(batches):
        walkers = sampler.sample(num_steps=steps_between).detach().requires_grad_(True)
        energy, _, _ = compute_local_energy(model, system, walkers, laplacian_method="exact")
        values = energy.detach().cpu().numpy()
        all_energies.append(values)
        block_means.append(float(values.mean()))
    merged = np.concatenate(all_energies)
    return {
        "samples": int(merged.size),
        "mean": float(merged.mean()),
        "std": float(merged.std(ddof=1)),
        "naive_sem": float(merged.std(ddof=1) / math.sqrt(merged.size)),
        "block_sem": float(np.std(block_means, ddof=1) / math.sqrt(len(block_means))),
        "block_means": block_means,
        "min": float(merged.min()),
        "max": float(merged.max()),
    }


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("This bounded audit run is configured for the requested CUDA Conda environment")
    device = "cuda"
    started = time.perf_counter()
    payload: dict[str, object] = {
        "environment": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "dtype": "float32",
        },
        "scope_note": "Short convergence diagnostics only; not a paper-level reproduction.",
        "h2": [],
        "lih": [],
    }
    save(payload)

    for clip_method in ("tanh", "none"):
        for seed in (0, 1, 42):
            run_started = time.perf_counter()
            result = train_vmc(
                system=build_h2_system(),
                num_epochs=100,
                num_walkers=128,
                mcmc_steps_per_epoch=10,
                burn_in_steps=100,
                lr=1e-3,
                clip_factor=5.0,
                clip_method=clip_method,
                laplacian_method="exact",
                grad_clip=1.0,
                device=device,
                dtype=torch.float32,
                seed=seed,
                verbose=False,
            )
            evaluation = evaluate_final_model(result, batches=5, steps_between=20)
            entry = {
                "seed": seed,
                "clip_method": clip_method,
                "epochs": 100,
                "reported_final_energy_pre_update_batch": result["final_energy"],
                "reported_final_variance": result["final_variance"],
                "history_first_energy": result["history"]["energy_raw"][0],
                "history_min_energy": min(result["history"]["energy_raw"]),
                "evaluation_after_final_update": evaluation,
                "elapsed_seconds": time.perf_counter() - run_started,
            }
            payload["h2"].append(entry)
            save(payload)
            del result
            torch.cuda.empty_cache()

    run_started = time.perf_counter()
    lih_result = train_vmc(
        system=build_lih_system(),
        num_epochs=20,
        num_walkers=128,
        mcmc_steps_per_epoch=10,
        burn_in_steps=100,
        lr=1e-3,
        clip_factor=5.0,
        clip_method="tanh",
        laplacian_method="hutchinson",
        num_hutchinson_samples=1,
        grad_clip=1.0,
        device=device,
        dtype=torch.float32,
        seed=42,
        verbose=False,
    )
    payload["lih"].append(
        {
            "seed": 42,
            "clip_method": "tanh",
            "laplacian_method": "hutchinson",
            "epochs": 20,
            "reported_final_energy_pre_update_batch": lih_result["final_energy"],
            "reported_final_variance": lih_result["final_variance"],
            "history_first_energy": lih_result["history"]["energy_raw"][0],
            "history_min_energy": min(lih_result["history"]["energy_raw"]),
            "evaluation_after_final_update": evaluate_final_model(lih_result, batches=3, steps_between=20),
            "elapsed_seconds": time.perf_counter() - run_started,
        }
    )
    payload["total_elapsed_seconds"] = time.perf_counter() - started
    save(payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
