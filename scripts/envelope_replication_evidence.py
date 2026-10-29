"""Immutable submission, timing gate and complete-data replication reporting."""

import copy
import json
import re
import shutil
import statistics
from pathlib import Path

import torch

from ferminet_pytorch.vmc_statistics import apply_reference, energy_statistics

from . import envelope_replication as replication
from .attribute_residual_energy import observations, verify_archive
from .factorial_campaign import seed_for
from .kaggle_campaign import file_sha, write_json
from .verify_kaggle_smoke import plan_digest


def project_budget(parents, cases, plan, batch_size):
    rates = {
        phase: max(statistics.mean(p["phase_step_seconds"][phase][5:]) for p in parents)
        for phase in ("frozen_jastrow", "joint_initial", "tail")
    }
    parent = plan["parent"]
    parent_seconds = (
        parent["jastrow_stage1_epochs"] * rates["frozen_jastrow"]
        + (parent["initial_epochs"] - parent["jastrow_stage1_epochs"])
        * rates["joint_initial"]
        + (parent["final_epochs"] - parent["initial_epochs"]) * rates["tail"]
        + max(p["burn_seconds"] for p in parents)
    )
    update = max(statistics.mean(c["step_seconds"][5:]) for c in cases)
    per_record = max(
        (
            c["evaluation"]["total_with_observations_seconds"]
            - c["evaluation"]["burn_in_seconds"]
        )
        / c["evaluation"]["time_records"]
        for c in cases
    )
    burn = max(c["evaluation"]["burn_in_seconds"] for c in cases)
    rethermalize = max(c["rethermalization_seconds"] for c in cases)
    # Parent generation is a barrier, then one arm per card per parent.
    seconds = parent_seconds + batch_size * (
        plan["training"]["num_epochs"] * update
        + rethermalize
        + (plan["evaluation"]["num_steps"] + plan["zero_evaluation_records"])
        * per_record
        + 2 * burn
    )
    projected = plan["timing_safety_factor"] * seconds + 900
    return {
        "seeds_per_batch": batch_size,
        "projected_seconds": projected,
        "within_budget": projected <= plan["max_projected_seconds"],
        "parent_update_seconds": rates,
        "child_update_seconds": update,
        "evaluation_seconds_per_record": per_record,
    }


def verify_parent(base, seed, plan):
    folder = base / "parents" / f"ferminet_seed{seed}"
    row = json.loads((folder / "result.json").read_text())
    state = torch.load(
        folder / "final_checkpoint.pt", weights_only=True, map_location="cpu"
    )
    first = torch.load(
        folder / "phase1_checkpoint.pt", weights_only=True, map_location="cpu"
    )
    expected = plan["parent"]["final_epochs"]
    stage = plan["parent"]["jastrow_stage1_epochs"]
    if (
        row["status"] != "completed"
        or state["epoch"] != expected
        or first["epoch"] != plan["parent"]["initial_epochs"]
        or len(state["history_rows"]) != expected
        or file_sha(folder / "final_checkpoint.pt") != row["checkpoint_sha256"]
        or file_sha(folder / "phase1_checkpoint.pt") != row["phase1_sha256"]
        or state["continuation"]["parent_checkpoint_sha256"] != row["phase1_sha256"]
        or not state["continuation"]["optimizer_state_preserved"]
        or any(
            float(v["step"]) != expected - stage
            for v in state["optimizer_state_dict"]["state"].values()
        )
        or state["sampler_settings"]["adapt_step_size"]
        or state["sampler_settings"]["total_steps"]
        != expected * 10 + plan["parent"]["burn_in_steps"]
        or not row["precision_after"]["passed"]
    ):
        raise ValueError("Parent state/lineage/update identity differs")
    return row


def verify_evaluation(folder, count, expected_seed):
    row = json.loads((folder / "result.json").read_text())
    energy = torch.load(folder / "energy.pt", weights_only=True, map_location="cpu")
    if (
        energy.shape != (count, 512)
        or energy.dtype != torch.float64
        or not torch.isfinite(energy).all()
    ):
        raise ValueError("Frozen energy shape/dtype differs")
    start = 0
    for shard in observations(folder / "observations", count, expected_seed):
        n = len(shard["energy"])
        if not torch.equal(shard["energy"], energy[start : start + n]):
            raise ValueError("Observation energy does not match time/walker trajectory")
        start += n
    computed = energy_statistics(energy)
    for key in ("energy_mean", "energy_sem", "energy_var", "tau_int", "split_rhat"):
        if abs(computed[key] - row[key]) > 1e-12:
            raise ValueError("Frozen statistics do not reproduce")
    if (
        row["seed"] != expected_seed
        or row["time_records"] != count
        or row["num_invalid"]
    ):
        raise ValueError("Evaluation seed/count differs")
    referenced = apply_reference(copy.deepcopy(computed), exact_energy=-1.174475714220)
    if referenced["chemical_accuracy_status"] != row["chemical_accuracy_status"]:
        raise ValueError("Chemical accuracy does not reproduce")
    return row


def verify_child(base, seed, arm, plan, parent):
    folder = base / f"ferminet_seed{seed}_{arm}"
    row = json.loads((folder / "result.json").read_text())
    state = torch.load(
        folder / "final_checkpoint.pt", weights_only=True, map_location="cpu"
    )
    count = plan["training"]["num_epochs"]
    if (
        row["status"] != "completed"
        or row["epochs"] != count
        or state["epoch"] != count
        or len(state["history_rows"]) != count
        or file_sha(folder / "final_checkpoint.pt") != row["checkpoint_sha256"]
        or state["experiment_origin"]["parent_checkpoint_sha256"]
        != parent["checkpoint_sha256"]
        or state["experiment_origin"]["parent_epoch"] != plan["parent"]["final_epochs"]
        or state["experiment_origin"]["optimizer_state_preserved"]
        or state["scheduler_state_dict"]["last_epoch"] != count
        or any(
            float(v["step"]) != count
            for v in state["optimizer_state_dict"]["state"].values()
        )
        or state["sampler_settings"]["adapt_step_size"]
        or state["sampler_settings"]["total_steps"] != count * 10 + 1000
        or state["model_identity"]["envelope_type"]
        != plan["arms"][arm]["envelope_type"]
        or state["model_identity"]["envelope_scale"] != 2.0
        or state["config"]["two_stage_jastrow"]
        or state["config"]["dtype"] != torch.float32
        or state["kinetic_mode"] != "reverse"
        or state["config"]["compile_sampler"]
        or state["loss_settings"]["clip_method"] != "tanh"
        or not row["precision_before"]["passed"]
        or not row["precision_after"]["passed"]
    ):
        raise ValueError("Child state/update/lineage differs")
    e = verify_evaluation(
        folder / "evaluation",
        plan["evaluation"]["num_steps"],
        seed_for(plan, seed, arm),
    )
    if e != row["evaluation"] or not e["extreme_precision_audit"]["passed"]:
        raise ValueError("Evaluation identity/audit differs")
    verify_evaluation(
        base / f"zero_seed{seed}_{plan['arms'][arm]['envelope_type']}",
        plan["zero_evaluation_records"],
        seed_for(
            plan, seed, zero=True, envelope_type=plan["arms"][arm]["envelope_type"]
        ),
    )
    return row


def verify_batch(download, revision, registered, smoke=False):
    base = Path(download) / "ferminet_results"
    campaign, artifacts = verify_archive(base, revision)
    plan = replication.smoke_plan(registered) if smoke else registered
    if campaign["plan"] != plan:
        raise ValueError("Downloaded protocol differs")
    environment = json.loads((base / "bootstrap.json").read_text())
    if (
        environment["source_commit"] != revision
        or len(environment["gpu_names"]) != 2
        or any("T4" not in g for g in environment["gpu_names"])
    ):
        raise ValueError("Two real T4 devices/source revision required")
    root = replication.ROOT
    for name, digest in campaign["source_sha256"].items():
        if file_sha(root / name) != digest:
            raise ValueError("Collector source differs from executed source: " + name)
    for device in range(2):
        log = (base / f"runtime_tests_gpu{device}.log").read_text()
        match = re.search(r"Ran (\d+) tests?", log)
        if (
            not match
            or int(match[1]) < 18
            or "\nOK\n" not in log.replace("\r\n", "\n")
            or re.search(r"skipped|FAILED|ERROR", log)
        ):
            raise ValueError("Runtime CUDA checks missing/failed/skipped")
    parents, rows = [], []
    for seed in campaign["seeds"]:
        parent = verify_parent(base, seed, plan)
        parents.append(parent)
        initial = []
        for arm in replication.ARMS:
            rows.append(verify_child(base, seed, arm, plan, parent))
            initial.append(
                torch.load(
                    base / f"ferminet_seed{seed}_{arm}/fork_initial_state.pt",
                    weights_only=True,
                    map_location="cpu",
                )
            )
        if any(not torch.equal(initial[0][k], initial[1][k]) for k in initial[0]):
            raise ValueError("A/C initial walkers/RNG differ")
        parent_state = torch.load(
            base / f"parents/ferminet_seed{seed}/final_checkpoint.pt",
            weights_only=True,
            map_location="cpu",
        )
        if not torch.equal(
            initial[0]["walkers"], parent_state["walkers"]
        ) or not torch.equal(initial[0]["rng"], parent_state["rng_state"]):
            raise ValueError("Fork differs from parent walkers/RNG")
    return campaign, artifacts, parents, rows


def gate(download, revision, plan, history):
    _campaign, artifacts, parents, rows = verify_batch(
        download, revision, plan, smoke=True
    )
    if len(parents) != 2 or len(rows) != 4:
        raise ValueError("Smoke requires two parents and four child cases")
    historical = set()
    for p in Path(history).rglob("result.json"):
        value = json.loads(p.read_text())
        if "seed" in value and "energy_mean" in value:
            historical.add(value["seed"])
    seeds = replication.check_seed_collisions(plan, historical)
    timings = [project_budget(parents, rows, plan, size) for size in (2, 1)]
    selected = next((t for t in timings if t["within_budget"]), None)
    return {
        "passed": selected is not None,
        "source_commit": revision,
        "long_plan_sha256": plan_digest(plan),
        "real_gpu_count": 2,
        "smoke_artifacts_verified": len(artifacts),
        "smoke_parents": 2,
        "smoke_child_cases": 4,
        "seeds_per_batch": selected["seeds_per_batch"] if selected else None,
        "timings": timings,
        "evaluation_seeds": seeds,
        "historical_evaluation_seed_count": len(historical),
        "scientific_configuration_selected_using_energy": False,
    }


def prepare(args, plan):
    if not args.revision or not re.fullmatch(r"[0-9a-f]{40}", args.revision):
        raise ValueError("Full immutable revision required")
    receipt = json.loads(args.gate.read_text()) if args.gate else None
    if not args.smoke and (
        not receipt
        or not receipt["passed"]
        or receipt["source_commit"] != args.revision
        or receipt["long_plan_sha256"] != plan_digest(plan)
    ):
        raise ValueError(
            "Formal submission requires this source/protocol's real T4 gate"
        )
    size = 2 if args.smoke else receipt["seeds_per_batch"]
    groups = replication.batches(
        replication.smoke_plan(plan) if args.smoke else plan, size
    )
    if not 0 <= args.batch_index < len(groups):
        raise ValueError("Invalid registered batch")
    slug = (
        "ferminet-ac-replication-smoke"
        if args.smoke
        else f"ferminet-ac-replication-b{args.batch_index + 1}"
    )
    arguments = ["--batch-index", str(args.batch_index), "--seeds-per-batch", str(size)]
    if args.smoke:
        arguments += ["--smoke"]
    preamble = (
        f'SOURCE_REPOSITORY = "https://github.com/spirlness/ferminet-pytorch.git"\n'
        f'SOURCE_COMMIT = {args.revision!r}\nCAMPAIGN_CONFIG = "configs/envelope_replication.json"\n'
        f'CAMPAIGN_MODULE = "scripts.envelope_replication"\nCAMPAIGN_ARGUMENTS = {arguments!r}\n'
        f"SMOKE_RUN = {args.smoke!r}\nSMOKE_GATE = {receipt!r}\n\n"
    )
    resume = getattr(args, "resume_kernel", None)
    if resume:
        if not re.fullmatch(r"[a-zA-Z0-9_-]+/[a-zA-Z0-9_-]+", resume):
            raise ValueError("Invalid resume kernel")
        preamble += f"RESUME_KERNEL = {resume!r}\n"
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "run.py").write_text(
        preamble + (replication.ROOT / "kaggle/bootstrap.py").read_text(),
        encoding="utf-8",
    )
    metadata = {
        "id": f"{args.owner}/{slug}",
        "title": slug.replace("-", " ").title(),
        "code_file": "run.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [resume] if resume else [],
    }
    write_json(args.output / "kernel-metadata.json", metadata)
    return metadata


def collect(downloads, output, revision, plan, history):
    if output.exists():
        raise ValueError("Preserve existing evidence: choose a new archive directory")
    rows, parents, sources = [], [], []
    for download in downloads:
        campaign, manifest, batch_parents, batch_rows = verify_batch(
            download, revision, plan
        )
        rows.extend(batch_rows)
        parents.extend(batch_parents)
        sources.append(
            {
                "path": str(download.resolve()),
                "artifacts": len(manifest),
                "batch": campaign["batch_index"],
            }
        )
    keys = [(r["seed"], r["arm"]) for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate batches/cases supplied")
    inference = replication.paired_effect(rows, plan)
    output.mkdir(parents=True)
    for download in downloads:
        base = download / "ferminet_results"
        batch = json.loads((base / "campaign.json").read_text())["batch_index"]
        for path in base.rglob("*"):
            if path.is_file() and path.suffix in (".json", ".log"):
                target = (
                    output / "artifacts" / f"batch_{batch}" / path.relative_to(base)
                )
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
    historical = json.loads(Path(history).read_text())["cases"] if history else []
    historical_ac = [r for r in historical if r["arm"] in replication.ARMS]
    combined = []
    for seed in sorted({r["seed"] for r in rows + historical_ac}):
        selected = {r["arm"]: r for r in rows + historical_ac if r["seed"] == seed}
        if set(selected) == set(replication.ARMS):
            combined.append(
                1000
                * (
                    selected["C"]["evaluation"]["energy_mean"]
                    - selected["A"]["evaluation"]["energy_mean"]
                )
            )
    summary = {
        "source_commit": revision,
        "protocol": plan,
        "downloads": sources,
        "parents": parents,
        "cases": rows,
        "primary": inference,
        "historical_cases": historical_ac,
        "exploratory_combined_mean_mha": statistics.mean(combined)
        if combined
        else None,
        "limits": "Only six new parents enter the primary test. Historical combination is exploratory.",
    }
    write_json(output / "verified_summary.json", summary)
    lines = [
        "# A/C 独立父种子复现",
        "",
        f"固定源码 `{revision}`；主要验证状态：**{inference['status']}**。",
        "",
        "| 种子 | 组别 | 能量 Ha | 绝对偏差 mHa | SEM mHa | 化学精度 | 诊断 |",
        "|---:|---|---:|---:|---:|---|---|",
    ]
    for row in sorted(rows, key=lambda r: (r["seed"], r["arm"])):
        e = row["evaluation"]
        lines.append(
            f"| {row['seed']} | {row['arm']} | {e['energy_mean']:.12f} | {e['abs_error_mha']:.6f} | {1000 * e['energy_sem']:.6f} | {e['chemical_accuracy_status']} | {e['diagnostics_status']} |"
        )
    if inference["effect"]:
        effect = inference["effect"]
        lines += [
            "",
            f"主要配对效应 C−A：{effect['mean_mha']:.6f} mHa，95% 近似区间 {effect['ci95_mha']}；预设改善判据通过：{effect['improvement_supported']}。",
        ]
    lines += [
        "",
        "旧三种子单列为历史证据；合并结果仅探索性展示，不替代主要验证。结论限于 Gaussian 父解迁移。缺组时不发布完整复现结论。",
    ]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(
        output / "archive_identity.json",
        {
            str(p.relative_to(output)): file_sha(p)
            for p in output.rglob("*")
            if p.is_file()
        },
    )
    return summary


def dispatch(args, plan):
    if args.command == "prepare":
        print(json.dumps(prepare(args, plan), indent=2))
    elif args.command == "gate":
        if not args.downloads or len(args.downloads) != 1 or not args.history:
            raise ValueError(
                "Gate requires one smoke download and historical evidence directory"
            )
        receipt = gate(args.downloads[0], args.revision, plan, args.history)
        write_json(args.output, receipt)
        print(json.dumps(receipt, indent=2))
        return 0 if receipt["passed"] else 2
    else:
        collect(args.downloads or [], args.output, args.revision, plan, args.history)
    return 0
