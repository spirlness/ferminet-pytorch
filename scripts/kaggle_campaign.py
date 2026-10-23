"""Independent seeds per GPU, bounded training, and held-out frozen evaluation.

No DDP or duplicated optimizer loop: workers call VMCDriver.step. All checkpoints
are selected at the predeclared end of training, never by evaluation energy.
"""

import argparse
import copy
import hashlib
import json
import math
import os
import platform
import signal
import statistics
import subprocess
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path

import torch

from ferminet_pytorch import FermiNet, TrainingConfig, VMCDriver
from ferminet_pytorch.evaluation import evaluate_frozen_wavefunction
from ferminet_pytorch.hamiltonian import compute_local_energy
from ferminet_pytorch.loss import VMCLoss
from ferminet_pytorch.mcmc import MCMCSampler
from ferminet_pytorch.systems import build_h2_system

from .campaign_logging import LoggedWorker, emit_event, gpu_snapshot
from .sampling_performance import configure_driver_sampling

STOP_REQUESTED = False


def request_stop(signum, frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True


class CaseProgress:
    def __init__(self, directory, interaction, seed, interval=10):
        self.directory = directory
        self.fields = dict(interaction=interaction, seed=seed, pid=os.getpid())
        self.interval, self.last_emit = interval, 0.0

    def update(self, status, force=False, event="progress", **fields):
        previous = self.fields.get("status")
        self.fields.update(status=status, **fields)
        now = time.monotonic()
        if force or status != previous or now - self.last_emit >= self.interval:
            from datetime import datetime, timezone

            self.fields["updated_utc"] = datetime.now(timezone.utc).isoformat()
            write_json(self.directory / "progress.json", self.fields)
            emit_event(
                event,
                interaction=self.fields["interaction"],
                seed=self.fields["seed"],
                pid=self.fields["pid"],
                status=status,
                **json_safe(fields),
            )
            self.last_emit = now


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, (torch.dtype, Path)):
        return str(value)
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(
        json.dumps(json_safe(value), indent=2, allow_nan=False), encoding="utf-8"
    )
    os.replace(temp, path)


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_identity():
    root = Path(__file__).resolve().parents[1]
    hashes = {
        str(p.relative_to(root)): file_sha(p)
        for folder in ("ferminet_pytorch", "scripts", "configs")
        for p in sorted((root / folder).rglob("*"))
        if p.is_file() and p.suffix in (".py", ".json")
    }
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    return {"commit": commit, "source_sha256": hashes}


def planned_cases(plan):
    if plan.get("schema_version") != 1:
        raise ValueError("Unsupported campaign schema")
    if not plan["seeds"] or len(set(plan["seeds"])) != len(plan["seeds"]):
        raise ValueError("Seeds must be nonempty and unique")
    if not plan["interactions"] or len(set(plan["interactions"])) != len(
        plan["interactions"]
    ):
        raise ValueError("Interactions must be nonempty and unique")
    if any(name not in ("ferminet", "psiformer") for name in plan["interactions"]):
        raise ValueError("Unknown interaction")
    if plan["training_dtype"] not in ("float32", "float64"):
        raise ValueError("Coordinate Hessians require float32 or float64, not AMP")
    if min(plan["training_seconds_per_case"], plan["checkpoint_seconds"]) <= 0:
        raise ValueError("Time budgets must be positive")
    if plan["training"].get("restore_best", True):
        raise ValueError(
            "Campaign evaluates the final checkpoint, not noisy best training energy"
        )
    return [
        (interaction, seed)
        for seed in plan["seeds"]
        for interaction in plan["interactions"]
    ]


def worker_cases(plan, index, count):
    cases = planned_cases(plan)
    if plan.get("balance_workers"):
        # Pair opposite orders on successive seeds to avoid a permanently
        # slower backbone owning an entire GPU. Summary order stays unchanged.
        cases = [
            (interaction, seed)
            for i, seed in enumerate(plan["seeds"])
            for interaction in (
                plan["interactions"][::-1] if i % 2 else plan["interactions"]
            )
        ]
    return cases[index::count]


def extend_case(driver, plan, interaction, seed):
    settings = plan["continuation"]
    parent = Path(os.environ["FERMINET_PARENT_OUTPUT"])
    case = f"{interaction}_seed{seed}"
    manifest = json.loads((parent / case / "manifest.json").read_text())
    result = json.loads((parent / case / "result.json").read_text())
    expected = settings["checkpoint_sha256"][case]
    if (
        manifest["commit"] != settings["source_commit"]
        or manifest["interaction"] != interaction
        or manifest["seed"] != seed
        or manifest["plan"]["model"] != plan["model"]
        or manifest["plan"]["training_dtype"] != plan["training_dtype"]
        or result["status"] != "completed"
        or result["epochs"] != settings["source_epochs"]
        or result["checkpoint_sha256"] != expected
    ):
        raise ValueError("Parent experiment identity/configuration does not match")
    driver.extend_from_checkpoint(
        parent / case / "final_checkpoint.pt", expected_sha256=expected
    )


def precision_audit(model, system, coordinates, settings):
    """Fixed points, same stored weights; not a global numerical error bound."""
    r = coordinates[: settings["num_points"]].detach().clone()
    high = copy.deepcopy(model).double()
    high_system = copy.deepcopy(system).to(dtype=torch.float64)
    with torch.no_grad():
        low_log, low_sign = model.forward_log_psi(r)
        high_log, high_sign = high.forward_log_psi(r.double())
    low_e, _, _ = compute_local_energy(model, system, r.requires_grad_(True))
    high_e, _, _ = compute_local_energy(
        high, high_system, r.detach().double().requires_grad_(True)
    )
    difference = low_e.detach().double() - high_e.detach()
    log_difference = float((low_log.double() - high_log).abs().max())
    rms = float(difference.square().mean().sqrt())
    passed = (
        bool(torch.isfinite(low_e).all() and torch.isfinite(high_e).all())
        and bool(torch.equal(low_sign, high_sign))
        and log_difference <= settings["max_log_difference"]
        and rms <= settings["max_energy_rms_difference"]
    )
    return dict(
        passed=passed,
        points=len(r),
        max_log_difference=log_difference,
        energy_rms_difference_ha=rms,
        max_energy_difference_ha=float(difference.abs().max()),
        scope="Fixed sampled points only; no universal error bound",
    )


def summarize(plan, output):
    cases = []
    for interaction, seed in planned_cases(plan):
        path = Path(output) / f"{interaction}_seed{seed}" / "result.json"
        progress = path.with_name("progress.json")
        row = dict(interaction=interaction, seed=seed, status="pending")
        if progress.exists():
            row.update(json.loads(progress.read_text()))
        if path.exists():
            row.update(json.loads(path.read_text()))
        cases.append(row)
    groups = {}
    for interaction in plan["interactions"]:
        rows = [
            c
            for c in cases
            if c["interaction"] == interaction and c["status"] == "completed"
        ]
        all_complete = len(rows) == len(plan["seeds"])
        group = dict(
            planned=len(plan["seeds"]), completed=len(rows), all_completed=all_complete
        )
        if rows:
            means = [c["evaluation"]["energy_mean"] for c in rows]
            sem_between = (
                statistics.stdev(means) / math.sqrt(len(means))
                if len(means) > 1
                else None
            )
            group.update(
                mean_of_checkpoint_energies=statistics.mean(means),
                min_energy=min(means),
                max_energy=max(means),
                between_seed_sem=sem_between,
                all_diagnostics_passed=all_complete
                and all(
                    c["evaluation"]["diagnostics_status"] == "passed" for c in rows
                ),
                chemical_accuracy_proven_for_all_seeds=all_complete
                and all(c["evaluation"]["chemical_accuracy_met"] for c in rows),
            )
            if len(rows) == 3 and all_complete:
                within = (
                    math.sqrt(sum(c["evaluation"]["energy_sem"] ** 2 for c in rows)) / 3
                )
                half = 4.302652729911275 * max(within, sem_between)
                mean = group["mean_of_checkpoint_energies"]
                group.update(
                    descriptive_seed_mean_ci95=[mean - half, mean + half],
                    interval_method="Student t df=2, max between/within SEM; not a pooled wavefunction",
                )
        groups[interaction] = group
    result = dict(
        cases=cases,
        seed_summary=groups,
        all_completed=all(c["status"] == "completed" for c in cases),
    )
    write_json(Path(output) / "summary.json", result)
    return result


def run_case(plan, interaction, seed, output, device):
    directory = Path(output) / f"{interaction}_seed{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    progress = CaseProgress(directory, interaction, seed, plan.get("log_seconds", 10))
    identity = dict(plan=plan, interaction=interaction, seed=seed, **source_identity())
    manifest = directory / "manifest.json"
    if manifest.exists() and json.loads(manifest.read_text()) != identity:
        raise ValueError("Resume requires the same plan, commit and source hashes")
    write_json(manifest, identity)
    result_path = directory / "result.json"
    if (
        result_path.exists()
        and json.loads(result_path.read_text())["status"] == "completed"
    ):
        return
    progress.update("initializing", event="case_start", device=device)
    torch.manual_seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats()
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
        interaction_type=interaction,
    ).to(device=device, dtype=dtype)
    config = TrainingConfig(
        **plan["training"], device=device, dtype=dtype, seed=seed, verbose=False
    )
    sampler = MCMCSampler(
        model,
        system,
        num_walkers=config.num_walkers,
        move_mode=config.move_mode,
        step_size=config.step_size,
    )
    driver = VMCDriver(
        model,
        system,
        sampler,
        config=config,
        loss_fn=VMCLoss(
            config.clip_factor, config.clip_method, max_invalid_fraction=0.0
        ),
    )
    checkpoint = directory / "resume.pt"
    if checkpoint.exists():
        driver.load_checkpoint(checkpoint)
        progress.update("resumed", epoch=driver.epoch)
    elif plan.get("continuation"):
        extend_case(driver, plan, interaction, seed)
        progress.update(
            "continued",
            force=True,
            event="continuation_loaded",
            epoch=driver.epoch,
            parent_sha256=driver.continuation["parent_checkpoint_sha256"],
            learning_rates=[g["lr"] for g in driver.optimizer.param_groups],
        )
    else:
        progress.update(
            "burn_in", burn_in_steps=0, requested_burn_in_steps=config.burn_in_steps
        )
        for step in range(0, config.burn_in_steps, 25):
            if STOP_REQUESTED:
                raise InterruptedError("Stop requested during burn-in")
            sampler.burn_in(min(25, config.burn_in_steps - step))
            progress.update(
                "burn_in", burn_in_steps=min(step + 25, config.burn_in_steps)
            )
    performance = None
    if plan.get("sampling_execution"):
        progress.update("sampling_benchmark", epoch=driver.epoch)
        performance_path = directory / "sampling_performance.json"
        mode = plan["sampling_execution"]
        if checkpoint.exists():
            mode = json.loads(performance_path.read_text())["selected"]
        driver, performance = configure_driver_sampling(
            driver,
            directory,
            plan.get("sampling_benchmark_horizon", config.num_epochs - driver.epoch),
            mode=mode,
        )
        write_json(performance_path, performance)
        emit_event(
            "sampling_execution", interaction=interaction, seed=seed, **performance
        )
    progress.update("precision_audit", epoch=driver.epoch)
    audit_before = precision_audit(
        model, system, sampler.walkers, plan["precision_audit"]
    )
    write_json(directory / "precision_before.json", audit_before)
    if not audit_before["passed"]:
        raise FloatingPointError("FP32/FP64 preflight failed; precision audit saved")
    origin = driver.continuation["parent_epoch"] if driver.continuation else 0
    previous_seconds = sum(row["epoch_time"] for row in driver.history[origin:])
    start, saved_at = time.perf_counter(), time.perf_counter()
    progress.update("training", epoch=driver.epoch, num_epochs=config.num_epochs)
    while driver.epoch < config.num_epochs:
        if STOP_REQUESTED:
            driver.save_checkpoint(checkpoint)
            progress.update("interrupted", force=True, epoch=driver.epoch)
            raise InterruptedError("Stop requested; training checkpoint saved")
        elapsed = previous_seconds + time.perf_counter() - start
        if elapsed >= plan["training_seconds_per_case"]:
            break
        row = driver.step()
        elapsed = previous_seconds + time.perf_counter() - start
        progress.update(
            "training",
            force=driver.epoch == origin + 1 or driver.epoch == config.num_epochs,
            event="training_step",
            epoch=driver.epoch,
            num_epochs=config.num_epochs,
            energy_mean=row["energy_mean"],
            energy_var=row["energy_var"],
            acceptance=row["mcmc_acceptance"],
            stage=driver.stage,
            step_seconds=row["epoch_time"],
            training_seconds=elapsed,
            learning_rates=[group["lr"] for group in driver.optimizer.param_groups],
            remaining_budget_seconds=max(
                0, plan["training_seconds_per_case"] - elapsed
            ),
            estimated_training_seconds_left=min(
                (config.num_epochs - driver.epoch) * row["epoch_time"],
                max(0, plan["training_seconds_per_case"] - elapsed),
            ),
        )
        if time.perf_counter() - saved_at >= plan["checkpoint_seconds"]:
            driver.save_checkpoint(checkpoint)
            progress.update(
                "training",
                force=True,
                event="checkpoint_saved",
                epoch=driver.epoch,
                checkpoint=str(checkpoint),
                checkpoint_epoch=driver.epoch,
            )
            saved_at = time.perf_counter()
    driver.save_checkpoint(checkpoint)
    driver.save_checkpoint(directory / "final_checkpoint.pt")
    progress.update(
        "checkpoint_saved",
        event="final_checkpoint_saved",
        epoch=driver.epoch,
        checkpoint=str(directory / "final_checkpoint.pt"),
        checkpoint_epoch=driver.epoch,
    )
    training_seconds = previous_seconds + time.perf_counter() - start
    write_json(directory / "training_history.json", driver.training_history)
    audit_after = precision_audit(
        model, system, sampler.walkers, plan["precision_audit"]
    )
    write_json(directory / "precision_after.json", audit_after)
    # Evaluate even when the post-training precision audit fails; preserve its
    # FP64 estimate, but never certify chemical accuracy for the failed audit.
    eval_model = copy.deepcopy(model).double()
    eval_system = copy.deepcopy(system).to(dtype=torch.float64)
    progress.update("evaluation_preparing", epoch=driver.epoch)
    eval_seed = (
        seed
        + (100000 if interaction == "ferminet" else 200000)
        + plan.get("evaluation_seed_offset", 0)
    )
    checkpoint_hash = file_sha(directory / "final_checkpoint.pt")
    evaluation = evaluate_frozen_wavefunction(
        eval_model,
        eval_system,
        **plan["evaluation"],
        device=device,
        dtype=torch.float64,
        seed=eval_seed,
        verbose=True,
        trajectory_path=str(directory / "evaluation_trajectory.pt"),
        progress_callback=lambda update: progress.update(**update),
        should_stop=lambda: STOP_REQUESTED,
    )
    if not audit_after["passed"]:
        evaluation["chemical_accuracy_met"] = False
        evaluation["chemical_accuracy_status"] = "numerical_precision_not_validated"
    state_after = file_sha(directory / "final_checkpoint.pt")
    if checkpoint_hash != state_after:
        raise RuntimeError("Training checkpoint changed during frozen evaluation")
    result = dict(
        status="completed",
        interaction=interaction,
        seed=seed,
        epochs=driver.epoch,
        updates_this_campaign=driver.epoch - origin,
        continuation=driver.continuation,
        sampling_performance=performance,
        training_seconds=training_seconds,
        training_stop="epochs" if driver.epoch == config.num_epochs else "time_budget",
        training_config=asdict(config),
        checkpoint_selection="final_no_energy_selection",
        checkpoint_sha256=checkpoint_hash,
        evaluation=evaluation,
        evaluation_dtype="float64, exact promotion of stored training weights",
        precision_before=audit_before,
        precision_after=audit_after,
        peak_cuda_memory_bytes=torch.cuda.max_memory_allocated()
        if device.startswith("cuda")
        else None,
    )
    result["device"] = dict(
        logical_device=device,
        visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        name=torch.cuda.get_device_name() if device.startswith("cuda") else "CPU",
    )
    write_json(result_path, result)
    progress.update(
        "completed",
        event="case_completed",
        force=True,
        epoch=driver.epoch,
        evaluation_energy=evaluation["energy_mean"],
        energy_sem=evaluation["energy_sem"],
        diagnostics=evaluation["diagnostics_status"],
        chemical_accuracy=evaluation["chemical_accuracy_status"],
    )


def worker(plan, output, index, count, device):
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    torch.set_num_threads(plan["cpu_threads_per_worker"])
    torch.set_float32_matmul_precision("highest")
    if device.startswith("cuda"):
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("Each worker must see exactly one CUDA device")
        torch.cuda.set_device(0)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    emit_event(
        "worker_start",
        worker=index,
        pid=os.getpid(),
        device=device,
        visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
    )
    for interaction, seed in worker_cases(plan, index, count):
        try:
            run_case(plan, interaction, seed, output, device)
        except Exception:
            status = "interrupted" if STOP_REQUESTED else "failed"
            write_json(
                Path(output) / f"{interaction}_seed{seed}" / "result.json",
                dict(
                    status=status,
                    interaction=interaction,
                    seed=seed,
                    traceback=traceback.format_exc(),
                ),
            )
            traceback.print_exc()
            emit_event(
                "case_failed",
                interaction=interaction,
                seed=seed,
                status=status,
                traceback=traceback.format_exc(),
            )
            return 130 if STOP_REQUESTED else 1
        finally:
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", type=int)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Local smoke only; never used by Kaggle",
    )
    args = parser.parse_args()
    plan = json.loads(args.config.read_text())
    planned_cases(plan)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.worker is not None:
        return worker(plan, args.output, args.worker, args.workers, args.device)
    if args.workers < 1:
        raise ValueError("Workers must be positive")
    if not args.allow_cpu and (
        not torch.cuda.is_available() or torch.cuda.device_count() < args.workers
    ):
        raise RuntimeError(
            f"Campaign requires {args.workers} real GPUs; no CPU fallback"
        )
    manifest = dict(
        plan=plan,
        **source_identity(),
        python=platform.python_version(),
        torch=str(torch.__version__),
        cuda=torch.version.cuda,
        gpu_names=[
            torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
        ],
    )
    write_json(args.output / "campaign.json", manifest)
    children = []
    for index in range(args.workers):
        env = os.environ.copy()
        env.update(
            CUDA_VISIBLE_DEVICES=str(index) if not args.allow_cpu else "",
            OMP_NUM_THREADS=str(plan["cpu_threads_per_worker"]),
            PYTHONUNBUFFERED="1",
        )
        command = [
            sys.executable,
            "-m",
            "scripts.kaggle_campaign",
            "--config",
            str(args.config.resolve()),
            "--output",
            str(args.output.resolve()),
            "--worker",
            str(index),
            "--workers",
            str(args.workers),
            "--device",
            "cpu" if args.allow_cpu else "cuda:0",
        ]
        children.append(
            LoggedWorker(command, args.output / f"worker_{index}.log", index, env=env)
        )
    started = time.monotonic()
    emit_event("campaign_start", workers=[child.status() for child in children])
    try:
        while any(child.process.poll() is None for child in children):
            summary = summarize(plan, args.output)
            emit_event(
                "heartbeat",
                cases=[
                    {
                        key: c.get(key)
                        for key in (
                            "interaction",
                            "seed",
                            "status",
                            "epoch",
                            "time_records",
                            "updated_utc",
                        )
                    }
                    for c in summary["cases"]
                ],
                workers=[child.status() for child in children],
                gpus=[] if args.allow_cpu else gpu_snapshot(),
            )
            if any(
                child.process.poll() not in (None, 0) or child.reader_error
                for child in children
            ):
                emit_event(
                    "campaign_failed",
                    reason="Worker exited unsuccessfully; stopping remaining workers",
                )
                break
            if time.monotonic() - started >= plan.get("max_wall_seconds", 34200):
                emit_event(
                    "campaign_failed", reason="Campaign wall-time budget exhausted"
                )
                break
            time.sleep(plan.get("heartbeat_seconds", 15))
    finally:
        for child in children:
            child.stop()
    summary = summarize(plan, args.output)
    emit_event(
        "campaign_finished",
        all_completed=summary["all_completed"],
        workers=[child.status() for child in children],
    )
    return (
        0
        if summary["all_completed"]
        and all(child.process.returncode == 0 for child in children)
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
