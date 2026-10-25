"""Preregistered paired clipping x envelope experiment using the sole VMC core."""

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

from ferminet_pytorch import FermiNet, TrainingConfig, VMCDriver
from ferminet_pytorch.evaluation import evaluate_frozen_wavefunction
from ferminet_pytorch.evaluation_observations import summarize_observations
from ferminet_pytorch.loss import VMCLoss
from ferminet_pytorch.mcmc import MCMCSampler
from ferminet_pytorch.systems import build_h2_system

from .campaign_logging import LoggedWorker, emit_event, gpu_snapshot
from .kaggle_campaign import file_sha, precision_audit, source_identity, write_json

STOP = False
ARMS = {
    "A": {"envelope_type": "gaussian", "clip_method": "tanh"},
    "B": {"envelope_type": "gaussian", "clip_method": "mean_hard"},
    "C": {"envelope_type": "soft_exponential", "clip_method": "tanh"},
    "D": {"envelope_type": "soft_exponential", "clip_method": "mean_hard"},
}


def request_stop(signum, frame):
    global STOP
    STOP = True


def case_id(seed, arm):
    return f"ferminet_seed{seed}_{arm}"


def schedule(plan):
    """Keep a shared zero-step evaluation on one worker; counterbalance order."""
    if (
        plan.get("experiment") != "clipping_envelope_factorial"
        or plan.get("schema_version") != 1
        or plan["arms"] != ARMS
        or plan["sampling_execution"] != "eager"
        or plan["envelope_scale"] != 2.0
    ):
        raise ValueError("Unsupported factorial plan")
    if (
        not plan["seeds"]
        or len(plan["seeds"]) != len(set(plan["seeds"]))
        or (not plan["smoke"] and plan["seeds"] != [11, 22, 33])
    ):
        raise ValueError("Factorial seeds differ from the preregistration")
    if not plan["smoke"] and (
        plan["training"]["num_epochs"] != 10000
        or plan["evaluation"]["num_steps"] != 2048
        or plan["zero_evaluation_records"] != 512
    ):
        raise ValueError("Formal training/evaluation budgets must not shrink")
    rng = random.Random(plan["schedule_seed"])
    queues = [[], []]
    for index, seed in enumerate(plan["seeds"]):
        gaussian_worker = index % 2
        for arm in ARMS:
            worker = gaussian_worker if arm in "AB" else 1 - gaussian_worker
            queues[worker].append(
                {"seed": seed, "arm": arm, "case_id": case_id(seed, arm)}
            )
    for queue in queues:
        rng.shuffle(queue)
    return queues


def seed_for(plan, seed, arm=None, zero=False, envelope_type=None):
    if zero:
        return (
            plan["zero_evaluation_seed_base"]
            + seed * 10
            + int(envelope_type == "soft_exponential")
        )
    return plan["evaluation_seed_base"] + seed * 10 + "ABCD".index(arm)


def make_driver(plan, seed, arm, device):
    options = ARMS[arm]
    dtype = getattr(torch, plan["training_dtype"])
    system = build_h2_system(device=device, dtype=dtype)
    model = FermiNet(
        system.nuclear_coords,
        system.nuclear_charges,
        1,
        1,
        **plan["model"],
        use_cusp_jastrow=True,
        spin_adapted=True,
        envelope_type=options["envelope_type"],
        envelope_scale=plan["envelope_scale"],
    ).to(device=device, dtype=dtype)
    config = TrainingConfig(
        **plan["training"],
        **options,
        envelope_scale=plan["envelope_scale"],
        use_cusp_jastrow=True,
        spin_adapted=True,
        smooth_cuspless=True,
        device=device,
        dtype=dtype,
        seed=seed,
        verbose=False,
    )
    sampler = MCMCSampler(
        model,
        system,
        num_walkers=config.num_walkers,
        device=device,
        dtype=dtype,
        move_mode=config.move_mode,
    )
    return VMCDriver(
        model,
        system,
        sampler,
        config=config,
        loss_fn=VMCLoss(
            config.clip_factor, config.clip_method, max_invalid_fraction=0.0
        ),
    )


def legacy_parent_identity(driver):
    # The pinned parent source assembled these factors unconditionally; its
    # historical TrainingConfig flags did not describe the resolved model.
    identity = driver._model_identity()
    identity.update(envelope_type="gaussian", envelope_scale=2.0)
    return identity


def eval_model(driver, plan, directory, seed, zero=False):
    directory.mkdir(parents=True, exist_ok=True)
    options = dict(plan["evaluation"])
    if zero:
        options["num_steps"] = plan["zero_evaluation_records"]
    model = copy.deepcopy(driver.model).double()
    evaluation_started = time.monotonic()
    system = copy.deepcopy(driver.system).to(dtype=torch.float64)
    progress_at = [0.0]

    def progress(fields):
        if STOP:
            raise InterruptedError("Stop requested")
        if time.monotonic() - progress_at[0] >= plan["log_seconds"]:
            emit_event("evaluation_progress", directory=directory.name, **fields)
            progress_at[0] = time.monotonic()

    devices = (
        [driver.sampler.device.index or 0]
        if driver.sampler.device.type == "cuda"
        else []
    )
    # Held-out evaluation never advances the formal training RNG.
    with torch.random.fork_rng(devices=devices):
        result = evaluate_frozen_wavefunction(
            model,
            system,
            **options,
            dtype=torch.float64,
            device=driver.sampler.device,
            seed=seed,
            verbose=False,
            trajectory_path=str(directory / "energy.pt"),
            observations_path=str(directory / "observations"),
            records_per_shard=plan["records_per_shard"],
            progress_callback=progress,
            should_stop=lambda: STOP,
        )
    result["tail_summary"] = summarize_observations(
        directory / "observations", system.nuclear_coords
    )
    positions = torch.tensor(
        result["tail_summary"].pop("extreme_audit_positions"),
        device=driver.sampler.device,
        dtype=driver.sampler.dtype,
    )
    result["extreme_precision_audit"] = precision_audit(
        driver.model,
        driver.system,
        positions,
        {
            "num_points": len(positions),
            "max_log_difference": 0.001,
            "max_energy_rms_difference": 0.001,
        },
    )
    result["total_with_observations_seconds"] = time.monotonic() - evaluation_started
    write_json(directory / "result.json", result)
    return result


def parent_path(plan, seed, parent_output):
    root = Path(parent_output)
    manifest = json.loads((root / "campaign.json").read_text())
    if manifest.get("commit") != plan["fork"]["source_commit"]:
        raise ValueError("Mounted parent source commit differs")
    parent_plan = manifest["plan"]
    if (
        parent_plan["model"] != plan["model"]
        or parent_plan["training"]["num_epochs"] != plan["fork"]["source_epochs"]
    ):
        raise ValueError("Mounted parent model/epoch differs")
    summary = json.loads((root / "summary.json").read_text())
    matches = [
        c
        for c in summary["cases"]
        if c["interaction"] == "ferminet" and c["seed"] == seed
    ]
    expected = plan["fork"]["checkpoint_sha256"][str(seed)]
    if (
        len(matches) != 1
        or matches[0]["status"] != "completed"
        or matches[0]["epochs"] != plan["fork"]["source_epochs"]
        or matches[0]["checkpoint_sha256"] != expected
    ):
        raise ValueError("Parent published checkpoint identity differs")
    target = root / f"ferminet_seed{seed}" / "final_checkpoint.pt"
    if file_sha(target) != expected:
        raise ValueError("Parent checkpoint SHA-256 mismatch")
    return target


def run_case(plan, case, output, parent_output, device):
    directory = output / case["case_id"]
    directory.mkdir(parents=True, exist_ok=True)
    emit_event(
        "case_start", **case, visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES")
    )
    manifest = dict(plan=plan, case=case, **source_identity())
    manifest_path = directory / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("Experiment identity differs from existing output")
    write_json(manifest_path, manifest)
    if (directory / "result.json").exists():
        saved = json.loads((directory / "result.json").read_text())
        if saved["status"] == "completed":
            return saved
    driver = make_driver(plan, case["seed"], case["arm"], device)
    parent = parent_path(plan, case["seed"], parent_output)
    checkpoint = directory / "resume.pt"
    init_seconds = 0.0
    rethermalization_seconds = 0.0
    if checkpoint.exists():
        driver.load_checkpoint(checkpoint)
    else:
        start = time.monotonic()
        driver.fork_from_checkpoint(
            parent,
            expected_sha256=plan["fork"]["checkpoint_sha256"][str(case["seed"])],
            parent_model_identity=legacy_parent_identity(driver),
        )
        before = precision_audit(
            driver.model,
            driver.system,
            driver.sampler.walkers[:32],
            {
                "num_points": 32,
                "max_log_difference": 0.001,
                "max_energy_rms_difference": 0.001,
            },
        )
        write_json(directory / "initial_audit.json", before)
        if not before["passed"]:
            raise FloatingPointError(
                "Fork numerical precision audit failed before training"
            )
        # Save the exact same pre-burn parent walkers/RNG identity for all arms.
        torch.save(
            {
                "walkers": driver.sampler.walkers.cpu(),
                "rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state()
                if device.startswith("cuda")
                else None,
            },
            directory / "fork_initial_state.pt",
        )
        zero_path = output / f"zero_seed{case['seed']}_{driver.model.envelope_type}"
        if not (zero_path / "result.json").exists():
            # A failed partial evaluation is retained under a separate attempt.
            if (zero_path / "observations").exists():
                zero_path.rename(
                    zero_path.with_name(zero_path.name + f"_partial_{time.time_ns()}")
                )
            eval_model(
                driver,
                plan,
                zero_path,
                seed_for(
                    plan,
                    case["seed"],
                    zero=True,
                    envelope_type=driver.model.envelope_type,
                ),
                zero=True,
            )
        burn_started = time.monotonic()
        for step in range(0, driver.config.burn_in_steps, 25):
            if STOP:
                raise InterruptedError("Stop during rethermalization")
            driver.sampler.burn_in(min(25, driver.config.burn_in_steps - step))
            emit_event(
                "burn_in",
                case_id=case["case_id"],
                burn_in_steps=min(step + 25, driver.config.burn_in_steps),
            )
        driver.sampler.adapt_step_size = False
        rethermalization_seconds = time.monotonic() - burn_started
        driver.save_checkpoint(checkpoint)
        init_seconds = time.monotonic() - start
    zero_path = output / f"zero_seed{case['seed']}_{driver.model.envelope_type}"
    start, last_log, last_save = time.monotonic(), 0.0, time.monotonic()
    while driver.epoch < driver.config.num_epochs:
        if STOP:
            driver.save_checkpoint(checkpoint)
            raise InterruptedError("Stop during training; checkpoint preserved")
        row = driver.step()
        now = time.monotonic()
        if (
            now - last_log >= plan["log_seconds"]
            or driver.epoch == driver.config.num_epochs
        ):
            emit_event(
                "training_step",
                case_id=case["case_id"],
                epoch=driver.epoch,
                energy_mean=row["energy_mean"],
                seconds_per_update=row["epoch_time"],
            )
            last_log = now
        if now - last_save >= plan["checkpoint_seconds"]:
            driver.save_checkpoint(checkpoint)
            last_save = now
    training_seconds = time.monotonic() - start
    driver.save_checkpoint(directory / "final_checkpoint.pt")
    driver.save_checkpoint(checkpoint)
    audit = precision_audit(
        driver.model,
        driver.system,
        driver.sampler.walkers[:32],
        {
            "num_points": 32,
            "max_log_difference": 0.001,
            "max_energy_rms_difference": 0.001,
        },
    )
    evaluation_dir = directory / "evaluation"
    if (evaluation_dir / "observations").exists():
        evaluation_dir.rename(directory / f"evaluation_partial_{time.time_ns()}")
    evaluation = eval_model(
        driver, plan, evaluation_dir, seed_for(plan, case["seed"], case["arm"])
    )
    result = dict(
        **case,
        status="completed",
        epochs=driver.epoch,
        cumulative_parent_plus_updates=plan["fork"]["source_epochs"] + driver.epoch,
        parent_checkpoint_sha256=driver.experiment_origin["parent_checkpoint_sha256"],
        model_identity=driver._model_identity(),
        precision_before=json.loads((directory / "initial_audit.json").read_text()),
        precision_after=audit,
        checkpoint_sha256=file_sha(directory / "final_checkpoint.pt"),
        zero_evaluation=str(zero_path.relative_to(output) / "result.json"),
        evaluation=evaluation,
        training_seconds=training_seconds,
        init_seconds=init_seconds,
        rethermalization_seconds=rethermalization_seconds,
        step_seconds=[r["epoch_time"] for r in driver.history],
        device={
            "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "name": torch.cuda.get_device_name()
            if device.startswith("cuda")
            else "cpu",
        },
    )
    write_json(directory / "result.json", result)
    emit_event("case_completed", case_id=case["case_id"], epochs=driver.epoch)
    return result


def paired_effects(cases, seeds, practical=0.2):
    """Approximate df=n-1 intervals; no pooling of different wavefunctions."""
    if (
        len(seeds) != 3
        or len(cases) != 12
        or any(c["status"] != "completed" for c in cases)
    ):
        return {"status": "incomplete", "effects": None}
    lookup = {(c["seed"], c["arm"]): c for c in cases}
    if len(lookup) != 12 or set(lookup) != {(s, a) for s in seeds for a in ARMS}:
        return {"status": "incomplete", "effects": None}
    if any(
        c["evaluation"]["diagnostics_status"] != "passed"
        or not c["precision_after"]["passed"]
        or not c["evaluation"]["extreme_precision_audit"]["passed"]
        for c in cases
    ):
        return {"status": "insufficient_evidence", "effects": None}
    # P(|T_2| > t) = 1 - t/sqrt(t^2+2); Bonferroni alpha=.05/3.
    alpha = 0.05 / 3
    critical = (1 - alpha) * math.sqrt(2 / (alpha * (2 - alpha)))
    effects = {}
    for name, weights in {
        "clipping": {"A": -0.5, "B": 0.5, "C": -0.5, "D": 0.5},
        "envelope": {"A": -0.5, "B": -0.5, "C": 0.5, "D": 0.5},
        "interaction": {"A": 1.0, "B": -1.0, "C": -1.0, "D": 1.0},
    }.items():
        values, mc_variances = [], []
        for seed in seeds:
            rows = {a: lookup[seed, a]["evaluation"] for a in ARMS}
            values.append(1000 * sum(weights[a] * rows[a]["energy_mean"] for a in ARMS))
            mc_variances.append(
                sum((1000 * weights[a] * rows[a]["energy_sem"]) ** 2 for a in ARMS)
            )
        mean = statistics.mean(values)
        seed_se = statistics.stdev(values) / math.sqrt(3)
        mc_se = math.sqrt(sum(mc_variances)) / 3
        se = max(seed_se, mc_se)
        interval = [mean - critical * se, mean + critical * se]
        effects[name] = {
            "paired_values_mha": values,
            "mean_mha": mean,
            "seed_sem_mha": seed_se,
            "monte_carlo_sem_mha": mc_se,
            "ci_bonferroni_mha": interval,
            "improvement_supported": mean <= -practical and interval[1] < 0,
        }
    return {
        "status": "complete",
        "effects": effects,
        "degrees_of_freedom": 2,
        "critical_value": critical,
        "family_alpha": 0.05,
        "limits": "Three optimized parent seeds, approximate intervals under heavy tails; interaction is nonadditivity, not an isolated energy gain.",
    }


def summarize(plan, output):
    rows = []
    for queue in schedule(plan):
        for case in queue:
            path = output / case["case_id"] / "result.json"
            rows.append(
                json.loads(path.read_text())
                if path.exists()
                else dict(**case, status="pending")
            )
    all_completed = all(c["status"] == "completed" for c in rows)
    result = {
        "cases": rows,
        "all_completed": all_completed,
        "inference": paired_effects(
            rows, plan["seeds"], plan["practical_improvement_mha"]
        )
        if not plan["smoke"]
        else {"status": "smoke_only", "effects": None},
    }
    write_json(output / "summary.json", result)
    return result


def worker(plan, output, parent_output, index, device):
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    torch.set_num_threads(plan["cpu_threads_per_worker"])
    torch.set_float32_matmul_precision("highest")
    if device.startswith("cuda"):
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("Worker must see exactly one GPU")
        torch.cuda.set_device(0)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    for case in schedule(plan)[index]:
        try:
            run_case(plan, case, output, parent_output, device)
        except Exception:  # noqa: BLE001 -- record any failed experiment before worker exit
            write_json(
                output / case["case_id"] / "result.json",
                dict(
                    **case,
                    status="interrupted" if STOP else "failed",
                    traceback=traceback.format_exc(),
                ),
            )
            traceback.print_exc()
            return 1
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--parent-output", default=os.environ.get("FERMINET_PARENT_OUTPUT")
    )
    parser.add_argument("--worker", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    plan = json.loads(args.config.read_text())
    queues = schedule(plan)
    if not args.parent_output or args.workers != 2:
        raise ValueError("Parent output and exactly two workers are required")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.worker is not None:
        return worker(plan, args.output, args.parent_output, args.worker, args.device)
    if not args.allow_cpu and (
        not torch.cuda.is_available() or torch.cuda.device_count() != 2
    ):
        raise RuntimeError("Exactly two real GPUs required")
    manifest = dict(plan=plan, schedule=queues, **source_identity())
    target = args.output / "campaign.json"
    if target.exists() and json.loads(target.read_text()) != manifest:
        raise ValueError("Campaign source/configuration differs")
    write_json(target, manifest)
    children = []
    for index in range(2):
        env = dict(
            os.environ,
            CUDA_VISIBLE_DEVICES=str(index) if not args.allow_cpu else "",
            PYTHONUNBUFFERED="1",
            OMP_NUM_THREADS="1",
        )
        children.append(
            LoggedWorker(
                [
                    sys.executable,
                    "-m",
                    "scripts.factorial_campaign",
                    "--config",
                    str(args.config.resolve()),
                    "--output",
                    str(args.output.resolve()),
                    "--parent-output",
                    str(Path(args.parent_output).resolve()),
                    "--worker",
                    str(index),
                    "--device",
                    "cpu" if args.allow_cpu else "cuda:0",
                ],
                args.output / f"worker_{index}.log",
                index,
                env=env,
            )
        )
    started = time.monotonic()
    try:
        while any(c.process.poll() is None for c in children):
            current = summarize(plan, args.output)
            emit_event(
                "heartbeat",
                completed=sum(c["status"] == "completed" for c in current["cases"]),
                workers=[c.status() for c in children],
                gpus=[] if args.allow_cpu else gpu_snapshot(),
            )
            if any(
                c.process.poll() not in (None, 0) or c.reader_error for c in children
            ):
                break
            if time.monotonic() - started >= plan["max_wall_seconds"]:
                emit_event("wall_budget_exhausted")
                break
            time.sleep(plan["heartbeat_seconds"])
    finally:
        for child in children:
            child.stop()
    result = summarize(plan, args.output)
    artifacts = {
        str(p.relative_to(args.output)): file_sha(p)
        for p in sorted(args.output.rglob("*"))
        if p.is_file() and p.name != "archive_manifest.json"
    }
    write_json(args.output / "archive_manifest.json", artifacts)
    return (
        0
        if result["all_completed"] and all(c.process.returncode == 0 for c in children)
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
