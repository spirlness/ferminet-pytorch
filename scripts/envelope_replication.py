"""Six independent Gaussian parents and preregistered paired A/C replication."""

import argparse
import copy
import json
import math
import os
import random
import signal
import statistics
import sys
import time
import traceback
from pathlib import Path

import torch

from . import factorial_campaign as factorial
from .campaign_logging import LoggedWorker, emit_event, gpu_snapshot
from .kaggle_campaign import file_sha, precision_audit, source_identity, write_json

ROOT = Path(__file__).resolve().parents[1]
SEEDS = [101, 202, 303, 404, 505, 606]
ARMS = {a: factorial.ARMS[a] for a in ("A", "C")}


def read_plan(path=None):
    plan = json.loads(
        Path(path or ROOT / "configs/envelope_replication.json").read_text()
    )
    expected = json.loads((ROOT / "configs/envelope_replication.json").read_text())
    if plan != expected or plan["seeds"] != SEEDS or plan["arms"] != ARMS:
        raise ValueError("Replication requires the registered protocol")
    return plan


def smoke_plan(plan):
    plan = copy.deepcopy(plan)
    plan.update(smoke=True, seeds=SEEDS[:2])
    plan["parent"].update(
        initial_epochs=200, final_epochs=400, jastrow_stage1_epochs=100
    )
    plan["training"]["num_epochs"] = 100
    plan["evaluation"]["num_steps"] = 64
    plan["zero_evaluation_records"] = 8
    plan["evaluation_seed_base"] = 23000000
    plan["zero_evaluation_seed_base"] = 24000000
    plan["max_wall_seconds"] = 2400
    return plan


def batches(plan, size=2):
    if size not in (1, 2):
        raise ValueError("Only preregistered one/two-parent batching is supported")
    return [plan["seeds"][i : i + size] for i in range(0, len(plan["seeds"]), size)]


def child_schedule(seeds, schedule_seed):
    queues = [[], []]
    for i, seed in enumerate(seeds):
        for arm in ARMS:
            worker = i % 2 if arm == "A" else 1 - i % 2
            queues[worker].append(
                {"seed": seed, "arm": arm, "case_id": factorial.case_id(seed, arm)}
            )
    rng = random.Random(schedule_seed)
    for queue in queues:
        rng.shuffle(queue)
    return queues


def check_seed_collisions(plan, historical):
    planned = {factorial.seed_for(plan, s, a) for s in plan["seeds"] for a in ARMS}
    planned |= {
        factorial.seed_for(plan, s, zero=True, envelope_type=e)
        for s in plan["seeds"]
        for e in ("gaussian", "soft_exponential")
    }
    if len(planned) != 4 * len(plan["seeds"]) or planned & set(historical):
        raise ValueError("Evaluation seeds overlap")
    return sorted(planned)


def paired_effect(rows, plan):
    expected = {(s, a) for s in SEEDS for a in ARMS}
    lookup = {(r["seed"], r["arm"]): r for r in rows}
    if (
        len(rows) != 12
        or set(lookup) != expected
        or any(r["status"] != "completed" for r in rows)
    ):
        return {"status": "incomplete", "effect": None}
    if any(
        r["evaluation"]["diagnostics_status"] != "passed"
        or not r["precision_after"]["passed"]
        or not r["evaluation"]["extreme_precision_audit"]["passed"]
        for r in rows
    ):
        return {"status": "insufficient_evidence", "effect": None}
    values, mc = [], []
    for seed in SEEDS:
        a, c = (lookup[seed, arm]["evaluation"] for arm in ARMS)
        values.append(1000 * (c["energy_mean"] - a["energy_mean"]))
        mc.append((1000 * a["energy_sem"]) ** 2 + (1000 * c["energy_sem"]) ** 2)
    seed_se = statistics.stdev(values) / math.sqrt(6)
    mc_se = math.sqrt(sum(mc)) / 6
    mean, se = statistics.mean(values), max(seed_se, mc_se)
    interval = [
        mean - plan["primary"]["critical_value"] * se,
        mean + plan["primary"]["critical_value"] * se,
    ]
    return {
        "status": "complete",
        "effect": {
            "contrast": "C-A",
            "paired_values_mha": values,
            "mean_mha": mean,
            "seed_sem_mha": seed_se,
            "monte_carlo_sem_mha": mc_se,
            "ci95_mha": interval,
            "degrees_of_freedom": 5,
            "improvement_supported": (mean <= -0.2 and interval[1] < 0),
            "limits": "Six new parents only; approximate interval; propagated MC errors assume independent evaluation streams.",
        },
    }


def parent_driver(plan, seed, device, tail=False):
    options = copy.deepcopy(plan)
    parent = plan["parent"]
    options["training"].update(
        num_epochs=parent["final_epochs"] if tail else parent["initial_epochs"],
        burn_in_steps=parent["burn_in_steps"],
        two_stage_jastrow=True,
        jastrow_stage1_epochs=parent["jastrow_stage1_epochs"],
        lr=1e-4 if tail else parent["initial_lr"],
        jastrow_lr=1e-5 if tail else parent["initial_jastrow_lr"],
        lr_schedule="warmup_inverse" if tail else "cosine",
    )
    return factorial.make_driver(options, seed, "A", device)


def train_until(driver, target, checkpoint, plan, phase):
    last_save, last_log = time.monotonic(), 0.0
    while driver.epoch < target:
        if factorial.STOP:
            driver.save_checkpoint(checkpoint)
            raise InterruptedError("Stop requested; checkpoint preserved")
        row = driver.step()
        now = time.monotonic()
        if now - last_log >= plan["log_seconds"] or driver.epoch == target:
            emit_event(
                "parent_training_step",
                seed=driver.config.seed,
                phase=phase,
                epoch=driver.epoch,
                seconds_per_update=row["epoch_time"],
            )
            last_log = now
        if now - last_save >= plan["checkpoint_seconds"]:
            driver.save_checkpoint(checkpoint)
            last_save = now
    driver.save_checkpoint(checkpoint)


def run_parent(plan, seed, output, device):
    folder = output / f"ferminet_seed{seed}"
    folder.mkdir(parents=True, exist_ok=True)
    identity = dict(plan=plan, seed=seed, **source_identity())
    manifest = folder / "manifest.json"
    if manifest.exists() and json.loads(manifest.read_text()) != identity:
        raise ValueError("Parent source/protocol identity differs")
    write_json(manifest, identity)
    if (folder / "result.json").exists():
        old = json.loads((folder / "result.json").read_text())
        if old["status"] == "completed":
            if file_sha(folder / "final_checkpoint.pt") != old["checkpoint_sha256"]:
                raise ValueError("Completed parent SHA differs")
            return old
    torch.manual_seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
    settings = {
        "num_points": 32,
        "max_log_difference": 0.001,
        "max_energy_rms_difference": 0.001,
    }
    phase1 = folder / "phase1_checkpoint.pt"
    if not phase1.exists():
        driver = parent_driver(plan, seed, device)
        resume = folder / "resume_initial.pt"
        if resume.exists():
            driver.load_checkpoint(resume)
        else:
            start = time.monotonic()
            for i in range(0, driver.config.burn_in_steps, 25):
                if factorial.STOP:
                    raise InterruptedError("Stop during initial parent burn-in")
                driver.sampler.burn_in(min(25, driver.config.burn_in_steps - i))
            driver.sampler.adapt_step_size = False
            write_json(folder / "burn.json", {"seconds": time.monotonic() - start})
            driver.save_checkpoint(resume)
        train_until(driver, plan["parent"]["initial_epochs"], resume, plan, "initial")
        driver.save_checkpoint(phase1)
        del driver
    driver = parent_driver(plan, seed, device, tail=True)
    resume = folder / "resume_tail.pt"
    if resume.exists():
        driver.load_checkpoint(resume)
    else:
        driver.extend_from_checkpoint(phase1, expected_sha256=file_sha(phase1))
        driver.save_checkpoint(resume)
    train_until(driver, plan["parent"]["final_epochs"], resume, plan, "tail")
    driver.save_checkpoint(folder / "final_checkpoint.pt")
    audit = precision_audit(
        driver.model, driver.system, driver.sampler.walkers, settings
    )
    if not audit["passed"]:
        write_json(folder / "precision_failure.json", audit)
        raise FloatingPointError("Parent precision audit failed")
    stage, initial = (
        plan["parent"][k] for k in ("jastrow_stage1_epochs", "initial_epochs")
    )
    times = [r["epoch_time"] for r in driver.history]
    result = {
        "interaction": "ferminet",
        "seed": seed,
        "status": "completed",
        "epochs": driver.epoch,
        "checkpoint_sha256": file_sha(folder / "final_checkpoint.pt"),
        "phase1_sha256": file_sha(phase1),
        "model_identity": driver._model_identity(),
        "precision_after": audit,
        "burn_seconds": json.loads((folder / "burn.json").read_text())["seconds"],
        "phase_step_seconds": {
            "frozen_jastrow": times[:stage],
            "joint_initial": times[stage:initial],
            "tail": times[initial:],
        },
        "continuation": driver.continuation,
    }
    write_json(folder / "result.json", result)
    emit_event("parent_completed", seed=seed, epoch=driver.epoch)
    return result


def derived_child_plan(plan, parent_output):
    manifest = json.loads((parent_output / "campaign.json").read_text())
    parents = json.loads((parent_output / "summary.json").read_text())["cases"]
    result = copy.deepcopy(plan)
    result["fork"] = {
        "source_commit": manifest["commit"],
        "source_epochs": plan["parent"]["final_epochs"],
        "checkpoint_sha256": {str(p["seed"]): p["checkpoint_sha256"] for p in parents},
    }
    return result


def summarize(plan, output):
    rows = []
    for seed in plan["seeds"]:
        for arm in ARMS:
            case = {"seed": seed, "arm": arm, "case_id": factorial.case_id(seed, arm)}
            path = output / case["case_id"] / "result.json"
            rows.append(
                json.loads(path.read_text())
                if path.exists()
                else dict(**case, status="pending")
            )
    result = {
        "cases": rows,
        "all_completed": all(r["status"] == "completed" for r in rows),
        "inference": {"status": "smoke_only", "effect": None}
        if plan["smoke"]
        else paired_effect(rows, plan),
    }
    write_json(output / "summary.json", result)
    return result


def worker(plan, seeds, output, phase, index, device):
    signal.signal(signal.SIGTERM, factorial.request_stop)
    signal.signal(signal.SIGINT, factorial.request_stop)
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    if device.startswith("cuda"):
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("Worker needs exactly one visible CUDA device")
        torch.cuda.set_device(0)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    try:
        if phase == "parent":
            for seed in seeds[index::2]:
                run_parent(plan, seed, output / "parents", device)
        else:
            child_plan = derived_child_plan(plan, output / "parents")
            for case in child_schedule(seeds, plan["schedule_seed"])[index]:
                factorial.run_case(child_plan, case, output, output / "parents", device)
        return 0
    except Exception:  # noqa: BLE001 -- preserve any scientific/worker failure
        write_json(
            output / f"failure_{phase}_{index}.json",
            {
                "status": "interrupted" if factorial.STOP else "failed",
                "traceback": traceback.format_exc(),
            },
        )
        traceback.print_exc()
        return 1


def run(plan, args):
    if not args.allow_cpu and (
        not torch.cuda.is_available() or torch.cuda.device_count() != 2
    ):
        raise RuntimeError("Two real CUDA devices required")
    seeds = batches(plan, args.seeds_per_batch)[args.batch_index]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    identity = dict(
        plan=plan,
        batch_index=args.batch_index,
        seeds_per_batch=args.seeds_per_batch,
        seeds=seeds,
        schedule=child_schedule(seeds, plan["schedule_seed"]),
        **source_identity(),
    )
    target = output / "campaign.json"
    if target.exists() and json.loads(target.read_text()) != identity:
        raise ValueError("Batch source/protocol/schedule differs")
    write_json(target, identity)
    started = time.monotonic()
    success = True
    for phase in ("parent", "children"):
        children = []
        for index in range(2):
            env = dict(
                os.environ,
                CUDA_VISIBLE_DEVICES="" if args.allow_cpu else str(index),
                PYTHONUNBUFFERED="1",
                OMP_NUM_THREADS="1",
                MKL_NUM_THREADS="1",
            )
            command = [
                sys.executable,
                "-u",
                "-m",
                "scripts.envelope_replication",
                "run",
                "--config",
                str(args.config.resolve()),
                "--output",
                str(output),
                "--batch-index",
                str(args.batch_index),
                "--seeds-per-batch",
                str(args.seeds_per_batch),
                "--phase",
                phase,
                "--worker",
                str(index),
                "--device",
                "cpu" if args.allow_cpu else "cuda:0",
            ]
            if args.smoke:
                command.append("--smoke")
            children.append(
                LoggedWorker(
                    command, output / f"{phase}_worker_{index}.log", index, env=env
                )
            )
        try:
            while any(c.process.poll() is None for c in children):
                emit_event(
                    "replication_heartbeat",
                    phase=phase,
                    seeds=seeds,
                    workers=[c.status() for c in children],
                    gpus=[] if args.allow_cpu else gpu_snapshot(),
                )
                if any(
                    c.process.poll() not in (None, 0) or c.reader_error
                    for c in children
                ):
                    success = False
                    break
                if time.monotonic() - started > plan["max_wall_seconds"] - 60:
                    success = False
                    emit_event("wall_budget_exhausted")
                    break
                time.sleep(plan["heartbeat_seconds"])
        finally:
            for child in children:
                child.stop()
        success &= all(
            c.process.returncode == 0 and not c.reader_error for c in children
        )
        if not success:
            break
        if phase == "parent":
            parent_root = output / "parents"
            parent_plan = copy.deepcopy(plan)
            parent_plan["training"]["num_epochs"] = plan["parent"]["final_epochs"]
            write_json(
                parent_root / "campaign.json",
                dict(plan=parent_plan, **source_identity()),
            )
            write_json(
                parent_root / "summary.json",
                {
                    "cases": [
                        json.loads(
                            (parent_root / f"ferminet_seed{s}/result.json").read_text()
                        )
                        for s in seeds
                    ]
                },
            )
    result = summarize(dict(plan, seeds=seeds), output)
    write_json(
        output / "archive_manifest.json",
        {
            str(p.relative_to(output)): file_sha(p)
            for p in sorted(output.rglob("*"))
            if p.is_file() and p.name != "archive_manifest.json"
        },
    )
    return 0 if success and result["all_completed"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        nargs="?",
        default="run",
        choices=["run", "prepare", "gate", "collect"],
    )
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/envelope_replication.json"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--batch-index", type=int, default=0)
    parser.add_argument("--seeds-per-batch", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--worker", type=int)
    parser.add_argument("--phase", choices=["parent", "children"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--revision")
    parser.add_argument("--gate", type=Path)
    parser.add_argument("--owner", default="hwbwhzbsh")
    parser.add_argument("--downloads", nargs="*", type=Path)
    parser.add_argument("--history", type=Path)
    parser.add_argument("--resume-kernel")
    args = parser.parse_args()
    plan = read_plan(args.config)
    if args.command != "run":
        from . import envelope_replication_evidence as evidence

        return evidence.dispatch(args, plan)
    if args.smoke:
        plan = smoke_plan(plan)
    if args.workers != 2:
        raise ValueError("Exactly two workers required")
    if args.worker is not None:
        seeds = batches(plan, args.seeds_per_batch)[args.batch_index]
        return worker(plan, seeds, args.output, args.phase, args.worker, args.device)
    return run(plan, args)


if __name__ == "__main__":
    raise SystemExit(main())
