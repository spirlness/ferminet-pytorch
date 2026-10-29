"""Audit and attribute saved frozen energies; never sample or update parameters."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import torch

from ferminet_pytorch import FermiNet

from .kaggle_campaign import file_sha, write_json

REFERENCE = -1.174475714220
OLD_SOURCE = "9b001c5bbd4c143058248bbc0c6e3585412ae27c"
FIELDS = ("positions", "energy", "kinetic", "potential", "log_abs", "sign")


def verify_archive(base, revision=None):
    campaign = json.loads((base / "campaign.json").read_text())
    if revision is not None and campaign["commit"] != revision:
        raise ValueError("Source revision differs")
    manifest = json.loads((base / "archive_manifest.json").read_text())
    for name, digest in manifest.items():
        if file_sha(base / name) != digest:
            raise ValueError("Archive SHA differs: " + name)
    return campaign, manifest


def observations(path, records, seed=None):
    manifest = json.loads((path / "manifest.json").read_text())
    if (
        not manifest["complete"]
        or manifest["time_records"] != records
        or manifest["num_walkers"] != 512
    ):
        raise ValueError("Incomplete observation manifest")
    if seed is not None and manifest["seed"] != seed:
        raise ValueError("Observation seed differs")
    start = 0
    for item in manifest["shards"]:
        target = path / item["file"]
        if file_sha(target) != item["sha256"]:
            raise ValueError("Observation SHA differs")
        data = torch.load(target, weights_only=True, map_location="cpu")
        count = item["records"]
        if (
            item["start_record"] != start
            or not torch.equal(data["time_index"], torch.arange(start, start + count))
            or not torch.equal(data["walker_id"], torch.arange(512))
        ):
            raise ValueError("Observation time/walker identity differs")
        for field in FIELDS:
            shape = (count, 512, 2, 3) if field == "positions" else (count, 512)
            if (
                data[field].shape != shape
                or data[field].dtype != torch.float64
                or not torch.isfinite(data[field]).all()
            ):
                raise ValueError("Malformed/nonfinite FP64 observation: " + field)
        if not torch.allclose(
            data["energy"], data["kinetic"] + data["potential"], atol=2e-12, rtol=0
        ):
            raise ValueError("E != T+V")
        if not ((data["sign"] == 1) | (data["sign"] == -1)).all():
            raise ValueError("Invalid saved wavefunction sign")
        yield data
        start += count
    if start != records:
        raise ValueError("Shard records differ")


def load_model(folder, row, device):
    path = folder / "final_checkpoint.pt"
    if file_sha(path) != row["checkpoint_sha256"]:
        raise ValueError("Checkpoint SHA differs")
    saved = torch.load(path, weights_only=True, map_location="cpu")
    identity = saved["model_identity"]
    if identity != row["model_identity"] or identity["spin_mode"] != "singlet":
        raise ValueError("Model identity differs")
    state = saved["model_state_dict"]
    model = FermiNet(
        state["nuclear_coords"].double(),
        state["nuclear_charges"].double(),
        1,
        1,
        **{
            k: identity[k]
            for k in (
                "num_determinants",
                "num_layers",
                "hidden_dim_one",
                "hidden_dim_two",
                "interaction_type",
                "envelope_type",
                "envelope_scale",
                "eps",
            )
        },
        use_cusp_jastrow=True,
        spin_adapted=True,
        smooth_cuspless=True,
    ).double()
    model.load_state_dict(state)
    return model.to(device).eval()


@torch.no_grad()
def geometry(model, positions):
    distance = torch.linalg.vector_norm(
        positions[:, :, None] - model.nuclear_coords[None, None], dim=-1
    )
    nearest = distance.min(-1).values
    up, down = model.forward_orbitals(torch.cat((positions, positions.flip(1))))
    terms = up[..., 0, 0] * down[..., 0, 0] * model.det_weights
    direct, swapped = terms.chunk(2)
    terms = torch.cat((direct, swapped), dim=1)
    scale = terms.abs().amax(1)
    scaled = terms / torch.where(scale > 0, scale, torch.ones_like(scale))[:, None]
    denominator = scaled.sum(1).abs()
    ratio = torch.where(
        denominator > 0,
        scaled.abs().sum(1) / denominator,
        torch.full_like(denominator, float("inf")),
    )
    far, near = nearest.max(1).values > 3, nearest.min(1).values < 0.05
    ee_distance = torch.linalg.vector_norm(positions[:, 0] - positions[:, 1], dim=-1)
    flags = (
        far.long()
        + 2 * near.long()
        + 4 * (ee_distance < 0.1).long()
        + 8 * (ratio > 1000).long()
    )
    log, sign = model.forward_log_psi(positions)
    return {
        "flags": flags,
        "log10_cancellation": ratio.log10(),
        "max_nearest": nearest.max(1).values,
        "min_nuclear": nearest.min(1).values,
        "electron_distance": ee_distance,
        "log_abs": log,
        "sign": sign,
        "zero_term_sum": (denominator == 0) | (scale == 0),
    }


def direct_kinetic(model, positions):
    """Independent explicit singlet amplitude, differentiating six coordinates."""
    r = positions.detach().clone().requires_grad_(True)

    def base(x):
        up, down = model.forward_orbitals(x)
        return (up[..., 0, 0] * down[..., 0, 0] * model.det_weights).sum(
            -1
        ) * model.jastrow(x, model.nuclear_coords).exp()

    amplitude = 0.5 * (base(r) + base(r.flip(1)))
    grad = torch.autograd.grad(amplitude.sum(), r, create_graph=True)[0].reshape(-1, 6)
    laplacian = torch.zeros_like(amplitude)
    for dimension in range(6):
        second = torch.autograd.grad(
            grad[:, dimension].sum(), r, retain_graph=dimension < 5
        )[0]
        laplacian += second.reshape(-1, 6)[:, dimension]
    return (-0.5 * laplacian / amplitude).detach()


def sufficient_statistics(energy, kinetic, potential, masks):
    """One row per mask: count, signed/positive/negative residual, E², T,V,T²,V²,TV."""
    residual = energy - REFERENCE
    values = torch.stack(
        (
            torch.ones_like(energy),
            residual,
            residual.clamp_min(0),
            residual.clamp_max(0),
            residual.square(),
            kinetic,
            potential,
            kinetic.square(),
            potential.square(),
            kinetic * potential,
        ),
        dim=-1,
    )
    return torch.stack([values[mask].sum(0) for mask in masks])


def summarize_sufficient(stats, total):
    n = float(total[0])
    mean = float(total[1] / n)
    denominator = float(total[4] - total[1].square() / n)
    rows = []
    for row in stats:
        count = float(row[0])
        rows.append(
            {
                "count": int(count),
                "probability": count / n,
                "conditional_energy_ha": float(row[1] / count + REFERENCE)
                if count
                else None,
                "signed_error_contribution_mha": float(1000 * row[1] / n),
                "positive_error_contribution_mha": float(1000 * row[2] / n),
                "negative_error_contribution_mha": float(1000 * row[3] / n),
                "variance_share": float(
                    (row[4] - 2 * mean * row[1] + mean * mean * count) / denominator
                )
                if denominator > 0
                else 0.0,
                "conditional_variance_ha2": float(
                    (row[4] - row[1].square() / count) / (count - 1)
                )
                if count > 1
                else None,
            }
        )
    return rows


def variance_decomposition(total):
    n = float(total[0])
    if n <= 1:
        return {
            "energy": 0.0,
            "kinetic": 0.0,
            "potential": 0.0,
            "twice_covariance": 0.0,
            "reconstruction_residual": 0.0,
        }
    variance = lambda square, mean: float(
        (total[square] - total[mean].square() / n) / (n - 1)
    )
    e, t, v = variance(4, 1), variance(7, 5), variance(8, 6)
    twice_covariance = float(2 * (total[9] - total[5] * total[6] / n) / (n - 1))
    return {
        "energy": e,
        "kinetic": t,
        "potential": v,
        "twice_covariance": twice_covariance,
        "reconstruction_residual": e - t - v - twice_covariance,
    }


def block_bootstrap(blocks, seed, repetitions=20000):
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randint(
        len(blocks), (repetitions, len(blocks)), generator=generator
    )
    draws = blocks[indices].sum(1)
    totals = draws.sum(1)
    mean = totals[:, 1] / totals[:, 0]
    denominator = totals[:, 4] - totals[:, 1].square() / totals[:, 0]
    contribution = 1000 * draws[:, :, 1] / totals[:, None, 0]
    numerator = (
        draws[:, :, 4]
        - 2 * mean[:, None] * draws[:, :, 1]
        + mean[:, None].square() * draws[:, :, 0]
    )
    shares = torch.where(
        denominator[:, None] > 0,
        numerator / denominator[:, None],
        torch.zeros_like(numerator),
    )
    rows = []
    for region in range(blocks.shape[1]):
        occupied = int((blocks[:, region, 0] > 0).sum())
        rows.append(
            {
                "occupied_blocks": occupied,
                "signed_error_ci95_mha": torch.quantile(
                    contribution[:, region],
                    torch.tensor([0.025, 0.975], dtype=torch.float64),
                ).tolist()
                if occupied >= 4
                else None,
                "variance_share_ci95": torch.quantile(
                    shares[:, region], torch.tensor([0.025, 0.975], dtype=torch.float64)
                ).tolist()
                if occupied >= 4
                else None,
                "status": "approximate_conditional"
                if occupied >= 4
                else "insufficient_occupied_blocks",
            }
        )
    return rows


def select_audit(energy, diagnostic, seed):
    residual = (energy.flatten() - REFERENCE).abs()
    top_e = residual.topk(min(32, len(residual))).indices
    cancellation = diagnostic["log10_cancellation"].flatten()
    top_c = cancellation.topk(min(32, len(residual))).indices
    ordinary = (residual <= 0.5) & (cancellation <= 3)
    distance = diagnostic["max_nearest"].flatten()
    generator = torch.Generator().manual_seed(seed)
    picked = []
    for lo, hi in zip([0, 1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6, float("inf")]):
        candidates = torch.where(ordinary & (distance >= lo) & (distance < hi))[0]
        picked.extend(
            candidates[
                torch.randperm(len(candidates), generator=generator)[:4]
            ].tolist()
        )
    ordinary_mask = ordinary.clone()
    ordinary_mask[picked] = False
    candidates = torch.where(ordinary_mask)[0]
    picked.extend(
        candidates[
            torch.randperm(len(candidates), generator=generator)[: 32 - len(picked)]
        ].tolist()
    )
    return torch.unique(
        torch.cat((top_e, top_c, torch.tensor(picked, dtype=torch.long)))
    ), {
        "largest_residual": top_e.tolist(),
        "largest_cancellation": top_c.tolist(),
        "stratified_ordinary": picked,
    }


def analyze_case(base, row, output, device, index):
    folder = base / row["case_id"]
    model = load_model(folder, row, device)
    evaluation = row["evaluation"]
    trajectory = torch.load(
        folder / "evaluation/energy.pt", weights_only=True, map_location="cpu"
    )
    diagnostic, blocks, kinetics, all_stats = {}, [], [], []
    offset, log_error, sign_mismatch, zero_count = 0, 0.0, 0, 0
    # Each published shard is exactly one registered 128-record time block.
    for data in observations(
        folder / "evaluation/observations", 2048, evaluation["seed"]
    ):
        if data["energy"].shape[0] != 128 or not torch.equal(
            data["energy"], trajectory[offset : offset + 128]
        ):
            raise ValueError("Registered time blocks / trajectory differ")
        geo_parts = {}
        for positions in data["positions"].reshape(-1, 2, 3).split(1024):
            values = geometry(model, positions.to(device))
            for key, value in values.items():
                geo_parts.setdefault(key, []).append(value.cpu())
        geo = {k: torch.cat(v) for k, v in geo_parts.items()}
        log_error = max(
            log_error, float((geo["log_abs"] - data["log_abs"].flatten()).abs().max())
        )
        sign_mismatch += int((geo["sign"] != data["sign"].flatten()).sum())
        zero_count += int(geo["zero_term_sum"].sum())
        for key in (
            "flags",
            "log10_cancellation",
            "max_nearest",
            "min_nuclear",
            "electron_distance",
        ):
            diagnostic.setdefault(key, []).append(geo[key].reshape(128, 512))
        e, t, v = (data[k].flatten() for k in ("energy", "kinetic", "potential"))
        masks = [geo["flags"] == i for i in range(16)]
        stats = sufficient_statistics(e, t, v, masks)
        blocks.append(stats)
        all_stats.append(
            sufficient_statistics(e, t, v, [torch.ones_like(e, dtype=torch.bool)])[0]
        )
        kinetics.append(data["kinetic"])
        offset += 128
    diagnostic = {k: torch.cat(v) for k, v in diagnostic.items()}
    block_tensor = torch.stack(blocks)
    region_stats, total = block_tensor.sum(0), torch.stack(all_stats).sum(0)
    regions = summarize_sufficient(region_stats, total)
    intervals = block_bootstrap(block_tensor, 22000000 + index)
    for i, row_region in enumerate(regions):
        row_region.update(bitmask=i, bootstrap=intervals[i])
    residual = trajectory.flatten() - REFERENCE
    centered = trajectory.flatten() - trajectory.mean()
    flag = diagnostic["flags"].flatten()
    overlapping = [(flag & bit) != 0 for bit in (1, 2, 4, 8)]
    distance = diagnostic["max_nearest"].flatten()
    radial = [
        (distance >= lo) & (distance < hi)
        for lo, hi in zip([0, 1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6, float("inf")])
    ]
    extreme = residual.abs() > 0.5
    top = torch.zeros_like(extreme)
    top[centered.abs().topk(max(1, int(len(centered) * 0.0001))).indices] = True
    torch.cat(kinetics).flatten()
    # V=E-T is not substituted here: read original saved V for independent decomposition above.
    mask_rows = []
    for mask in overlapping + radial + [extreme, top]:
        values = residual[mask]
        mask_rows.append(
            {
                "count": int(mask.sum()),
                "probability": float(mask.double().mean()),
                "signed_error_contribution_mha": float(
                    1000 * values.sum() / len(residual)
                ),
                "positive_error_contribution_mha": float(
                    1000 * values.clamp_min(0).sum() / len(residual)
                ),
                "negative_error_contribution_mha": float(
                    1000 * values.clamp_max(0).sum() / len(residual)
                ),
                "variance_share": float(
                    centered[mask].square().sum() / centered.square().sum()
                )
                if float(centered.square().sum())
                else 0.0,
            }
        )
    selected, selection = select_audit(trajectory, diagnostic, 22500000 + index)
    selected_positions, selected_kinetic = [], []
    start = 0
    for data in observations(
        folder / "evaluation/observations", 2048, evaluation["seed"]
    ):
        size = data["energy"].numel()
        local = selected[(selected >= start) & (selected < start + size)] - start
        selected_positions.append(data["positions"].reshape(-1, 2, 3)[local])
        selected_kinetic.append(data["kinetic"].flatten()[local])
        start += size
    positions = torch.cat(selected_positions)
    stored = torch.cat(selected_kinetic)
    direct = torch.cat(
        [direct_kinetic(model, x.to(device)).cpu() for x in positions.split(16)]
    )
    finite = torch.isfinite(direct)
    passed = bool(finite.all() and torch.allclose(direct, stored, atol=1e-7, rtol=1e-8))
    decomposition = variance_decomposition(total)
    reconstructed = sum(r["signed_error_contribution_mha"] for r in regions)
    if abs(reconstructed - 1000 * float(trajectory.mean() - REFERENCE)) > 1e-10:
        raise ValueError("Disjoint contributions do not reconstruct mean error")
    if abs(decomposition["reconstruction_residual"]) > 1e-9:
        raise ValueError("Variance decomposition does not reconstruct")
    output.mkdir(parents=True, exist_ok=False)
    torch.save(
        dict(**diagnostic, time_index=torch.arange(2048), walker_id=torch.arange(512)),
        output / "diagnostics.pt",
    )
    torch.save(
        {
            "flat_index": selected,
            "positions": positions,
            "stored_kinetic": stored,
            "direct_kinetic": direct,
        },
        output / "kinetic_audit.pt",
    )
    result = {
        "case_id": row["case_id"],
        "seed": row["seed"],
        "arm": row["arm"],
        "source_checkpoint_sha256": row["checkpoint_sha256"],
        "regions": regions,
        "overlapping": dict(
            zip(["far", "near_nucleus", "near_electron", "cancellation"], mask_rows[:4])
        ),
        "radial_bins": dict(
            zip(["0_1", "1_2", "2_3", "3_4", "4_5", "5_6", "6_inf"], mask_rows[4:11])
        ),
        "extreme": mask_rows[-2],
        "top_0_01_percent": mask_rows[-1],
        "variance_decomposition": decomposition,
        "energy_mean": float(trajectory.mean()),
        "signed_error_mha": 1000 * float(trajectory.mean() - REFERENCE),
        "log_max_error": log_error,
        "sign_mismatches": sign_mismatch,
        "exact_zero_term_sums": zero_count,
        "kinetic_audit": {
            "passed": passed,
            "num_points": len(selected),
            "nonfinite": int((~finite).sum()),
            "max_abs_difference": float((direct - stored).abs().max())
            if finite.all()
            else None,
            "selection": selection,
        },
        "all_audits_passed": passed
        and log_error < 1e-8
        and sign_mismatch == 0
        and zero_count == 0,
        "limits": "Exploratory attribution conditional on visited trajectories. Cancellation is a proxy, not node distance or causality.",
    }
    write_json(output / "result.json", result)
    print(
        json.dumps(
            {
                "event": "residual_attribution_completed",
                "case_id": row["case_id"],
                "audits": result["all_audits_passed"],
            }
        ),
        flush=True,
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    torch.set_num_threads(1)
    base = args.input.resolve() / "ferminet_results"
    campaign, manifest = verify_archive(base, OLD_SOURCE)
    for name, digest in campaign["source_sha256"].items():
        data = subprocess.check_output(["git", "show", f"{OLD_SOURCE}:{name}"])
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError("Historical source bytes differ")
    identity = {
        "source_commit": OLD_SOURCE,
        "archive_manifest_sha256": file_sha(base / "archive_manifest.json"),
        "protocol": "16-bitmask-regions-v1",
        "block_records": 128,
        "bootstrap_repetitions": 20000,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    protocol = args.output / "identity.json"
    if protocol.exists() and json.loads(protocol.read_text()) != identity:
        raise ValueError("Attribution input identity differs")
    write_json(protocol, identity)
    rows = json.loads((base / "summary.json").read_text())["cases"]
    if len(rows) != 12 or any(r["status"] != "completed" for r in rows):
        raise ValueError("All 12 historical cases required")
    results = []
    for index, row in enumerate(sorted(rows, key=lambda r: r["case_id"])):
        folder = args.output / row["case_id"]
        if folder.exists():
            if not (folder / "result.json").exists():
                raise ValueError(
                    "Partial attribution directory; preserve and use a new output"
                )
            result = json.loads((folder / "result.json").read_text())
            if result["source_checkpoint_sha256"] != row["checkpoint_sha256"]:
                raise ValueError("Attribution resume identity differs")
        else:
            result = analyze_case(base, row, folder, args.device, index)
        results.append(result)
    zero = {
        p.parent.name: json.loads(p.read_text())
        for p in base.glob("zero_*/result.json")
    }
    write_json(
        args.output / "summary.json",
        {
            "identity": identity,
            "cases": results,
            "zero_evaluations_descriptive": zero,
            "all_audits_passed": all(r["all_audits_passed"] for r in results),
            "source_artifacts_verified": len(manifest),
        },
    )
    write_json(
        args.output / "archive_manifest.json",
        {
            str(p.relative_to(args.output)): file_sha(p)
            for p in sorted(args.output.rglob("*"))
            if p.is_file() and p.name != "archive_manifest.json"
        },
    )


if __name__ == "__main__":
    main()
