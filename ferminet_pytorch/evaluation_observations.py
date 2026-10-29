"""Bounded, indexed shards of frozen observations and spatial tail summaries."""

import hashlib
import json
import os
from pathlib import Path

import torch


class ObservationWriter:
    def __init__(self, path, records_per_shard, num_walkers, seed):
        self.path = Path(path)
        if records_per_shard < 1:
            raise ValueError("records_per_shard must be positive")
        self.path.mkdir(parents=True, exist_ok=True)
        if any(self.path.iterdir()):
            raise ValueError("Observation directory must be empty")
        self.limit, self.walkers, self.seed = records_per_shard, num_walkers, seed
        self.pending, self.shards, self.records = [], [], 0

    def append(self, **values):
        for value in values.values():
            if value.shape[0] != self.walkers or not torch.isfinite(value).all():
                raise FloatingPointError("Nonfinite or malformed frozen observation")
        self.pending.append({k: v.detach().cpu() for k, v in values.items()})
        if len(self.pending) >= self.limit:
            self.flush()

    def flush(self):
        if not self.pending:
            return
        count = len(self.pending)
        payload = {
            k: torch.stack([v[k] for v in self.pending]) for k in self.pending[0]
        }
        payload.update(
            time_index=torch.arange(self.records, self.records + count),
            walker_id=torch.arange(self.walkers),
        )
        name = f"observations_{len(self.shards):04d}.pt"
        target = self.path / name
        torch.save(payload, target.with_suffix(".tmp"))
        os.replace(target.with_suffix(".tmp"), target)
        self.shards.append(
            {
                "file": name,
                "start_record": self.records,
                "records": count,
                "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            }
        )
        self.records += count
        self.pending.clear()

    def close(self, complete):
        self.flush()
        manifest = {
            "schema_version": 1,
            "complete": complete,
            "seed": self.seed,
            "time_records": self.records,
            "num_walkers": self.walkers,
            "axes": "time,walker[,electron,xyz]",
            "shards": self.shards,
        }
        target = self.path / "manifest.json"
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        os.replace(temporary, target)
        return str(target.resolve())


def summarize_observations(path, nuclear_coords, reference=-1.174475714220):
    """Describe actual sampled tails without removing or clipping observations."""
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text())
    if not manifest["complete"]:
        raise ValueError("Tail summary requires complete observations")
    energy_parts = []
    expected = 0
    for shard in manifest["shards"]:
        target = path / shard["file"]
        if hashlib.sha256(target.read_bytes()).hexdigest() != shard["sha256"]:
            raise ValueError("Observation shard hash mismatch")
        data = torch.load(target, weights_only=True)
        count = shard["records"]
        if (
            shard["start_record"] != expected
            or not torch.equal(
                data["time_index"], torch.arange(expected, expected + count)
            )
            or not torch.equal(data["walker_id"], torch.arange(manifest["num_walkers"]))
        ):
            raise ValueError("Observation time/walker identity differs")
        energy_parts.append(data["energy"].double().reshape(-1))
        expected += count
    energy = torch.cat(energy_parts)
    if expected != manifest["time_records"]:
        raise ValueError("Observation manifest is incomplete")
    mean = energy.mean()
    denominator = float((energy - mean).square().sum())
    top_count = max(1, int(energy.numel() * 0.0001))
    top_indices = (energy - mean).abs().topk(top_count).indices
    top_mask = torch.zeros(len(energy), dtype=torch.bool)
    top_mask[top_indices] = True
    regions = {
        name: {
            "count": 0,
            "extreme_count": 0,
            "squared_deviation": 0.0,
            "signed_error_sum": 0.0,
            "top_count": 0,
        }
        for name in (
            "far_nearest_nucleus_gt_3",
            "near_nucleus_lt_0.05",
            "near_electron_lt_0.1",
        )
    }
    offset, extreme_count = 0, 0
    extreme_positions = []
    audit_indices = (energy - mean).abs().topk(min(32, len(energy))).indices
    for shard in manifest["shards"]:
        data = torch.load(path / shard["file"], weights_only=True)
        positions = data["positions"].double().reshape(-1, *data["positions"].shape[2:])
        values = data["energy"].double().reshape(-1)
        if not torch.isfinite(values).all():
            raise FloatingPointError("Nonfinite saved energy")
        distances = torch.linalg.vector_norm(
            positions[:, :, None] - nuclear_coords.double().cpu()[None, None], dim=-1
        )
        far = distances.min(-1).values.max(-1).values > 3
        near = distances.amin((-1, -2)) < 0.05
        if positions.shape[1] > 1:
            i, j = torch.triu_indices(positions.shape[1], positions.shape[1], 1)
            near_ee = (
                torch.linalg.vector_norm(positions[:, i] - positions[:, j], dim=-1)
                .min(-1)
                .values
                < 0.1
            )
        else:
            near_ee = torch.zeros(len(values), dtype=torch.bool)
        extreme = (values - reference).abs() > 0.5
        extreme_count += int(extreme.sum())
        for name, mask in zip(regions, (far, near, near_ee)):
            row = regions[name]
            row["count"] += int(mask.sum())
            row["extreme_count"] += int((mask & extreme).sum())
            row["squared_deviation"] += float((values[mask] - mean).square().sum())
            row["signed_error_sum"] += float((values[mask] - reference).sum())
            row["top_count"] += int(
                (mask & top_mask[offset : offset + len(values)]).sum()
            )
        local = (
            audit_indices[
                (audit_indices >= offset) & (audit_indices < offset + len(values))
            ]
            - offset
        )
        extreme_positions.extend(positions[local].tolist())
        offset += len(values)
    for row in regions.values():
        row["variance_share"] = (
            row.pop("squared_deviation") / denominator if denominator else 0.0
        )
        row["signed_error_contribution_mha"] = (
            1000 * row.pop("signed_error_sum") / len(energy)
        )
    return {
        "total_samples": len(energy),
        "energy_mean": float(mean),
        "regions_overlap": True,
        "regions": regions,
        "extreme_threshold_abs_error_ha": 0.5,
        "extreme_count": extreme_count,
        "top_count": top_count,
        "top_variance_share": float((energy[top_indices] - mean).square().sum())
        / denominator
        if denominator
        else 0.0,
        "extreme_audit_positions": extreme_positions,
        "limits": "Spatial flags overlap; variance attribution does not establish a causal mechanism.",
    }
