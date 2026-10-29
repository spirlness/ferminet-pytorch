"""Reproduce postanalysis from downloaded trajectories without new sampling.

Run from the repository root with a Python containing NumPy and PyTorch.
The input directory is campaign_runs/continue_diagnosis_20261023.
"""

import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from ferminet_pytorch.loss import clip_local_energy


def main():
    inputs = ROOT / "campaign_runs/continue_diagnosis_20261023"
    output = inputs / "reproduced"
    output.mkdir(exist_ok=True)
    rng = np.random.default_rng(20261023)
    rows = []
    for path in sorted(inputs.glob("ferminet_results/*/evaluation_trajectory.pt")):
        x = torch.load(path, map_location="cpu", weights_only=True).numpy()
        mean = float(x.mean())
        time_means = x.mean(1)
        blocks = time_means.reshape(16, 128).mean(1)
        bootstrap = blocks[rng.integers(0, 16, size=(20000, 16))].mean(1)
        interval = np.quantile(bootstrap, [.025, .975])
        half = x.shape[0] // 2
        deviation = (x - mean) ** 2
        rows.append({
            "case": path.parent.name,
            "block_size": 128,
            "block_count": 16,
            "bootstrap_replicates": 20000,
            "block_bootstrap_percentile_ci95_Ha": interval.tolist(),
            "signed_error_ci95_mHa": ((interval + 1.174475714220) * 1000).tolist(),
            "half_mean_difference_mHa": float((x[half:].mean() - x[:half].mean()) * 1000),
            "energy_below_minus2_count": int((x < -2).sum()),
            "energy_below_minus2_variance_share": float(deviation[x < -2].sum() / deviation.sum()),
            "energy_above_zero_count": int((x > 0).sum()),
            "eight_segment_means": x.reshape(8, 256, 512).mean((1, 2)).tolist(),
            "limits": "Conditional empirical block bootstrap cannot account for unsampled tails/modes; only 16 blocks; no new observations.",
        })
    (output / "tail_uncertainty.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    archived = json.loads((Path(__file__).parent / "tail_uncertainty.json").read_text())
    assert rows == archived, "Postanalysis differs from the archived result"

    energies = torch.tensor([-1.190, -1.176, -1.175, -1.174, -1.173, -1.172, -1.150], dtype=torch.float64)
    a, _, _ = clip_local_energy(energies)
    b, _, _ = clip_local_energy(energies + 100)
    delta = float(((a - a.mean()) - (b - b.mean())).abs().max())
    archived = json.loads((Path(__file__).parent / "energy_shift_invariance.json").read_text())
    assert delta == archived["max_difference_Ha"]
    print("Six trajectory summaries and the energy-shift counterexample reproduced.")


if __name__ == "__main__":
    main()
