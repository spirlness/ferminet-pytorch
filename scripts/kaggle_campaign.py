"""Independent seeds per GPU, bounded training, and held-out frozen evaluation.

No DDP or duplicated optimizer loop: workers call VMCDriver.step. All checkpoints
are selected at the predeclared end of training, never by evaluation energy.
"""

import argparse
import copy
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
import traceback

import torch

from ferminet_pytorch import FermiNet, TrainingConfig, VMCDriver
from ferminet_pytorch.evaluation import evaluate_frozen_wavefunction
from ferminet_pytorch.hamiltonian import compute_local_energy
from ferminet_pytorch.loss import VMCLoss
from ferminet_pytorch.mcmc import MCMCSampler
from ferminet_pytorch.systems import build_h2_system


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
        cases.append(
            json.loads(path.read_text())
            if path.exists()
            else dict(interaction=interaction, seed=seed, status="missing")
        )
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
    else:
        sampler.burn_in(config.burn_in_steps)
    audit_before = precision_audit(
        model, system, sampler.walkers, plan["precision_audit"]
    )
    write_json(directory / "precision_before.json", audit_before)
    if not audit_before["passed"]:
        raise FloatingPointError("FP32/FP64 preflight failed; precision audit saved")
    previous_seconds = sum(row["epoch_time"] for row in driver.history)
    start, saved_at = time.perf_counter(), time.perf_counter()
    while driver.epoch < config.num_epochs:
        elapsed = previous_seconds + time.perf_counter() - start
        if elapsed >= plan["training_seconds_per_case"]:
            break
        row = driver.step()
        if driver.epoch % 100 == 0:
            print(
                f"{interaction}/{seed} epoch={driver.epoch} E={row['energy_mean']:.8f} stage={driver.stage}",
                flush=True,
            )
        if time.perf_counter() - saved_at >= plan["checkpoint_seconds"]:
            driver.save_checkpoint(checkpoint)
            write_json(
                directory / "progress.json",
                dict(
                    epoch=driver.epoch,
                    training_seconds=elapsed,
                    last_metrics=row,
                    status="training",
                ),
            )
            saved_at = time.perf_counter()
    driver.save_checkpoint(checkpoint)
    driver.save_checkpoint(directory / "final_checkpoint.pt")
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
    write_json(
        directory / "progress.json", dict(epoch=driver.epoch, status="evaluating")
    )
    eval_seed = seed + (100000 if interaction == "ferminet" else 200000)
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


def worker(plan, output, index, count, device):
    torch.set_num_threads(plan["cpu_threads_per_worker"])
    torch.set_float32_matmul_precision("highest")
    if device.startswith("cuda"):
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("Each worker must see exactly one CUDA device")
        torch.cuda.set_device(0)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    failed = False
    for interaction, seed in planned_cases(plan)[index::count]:
        try:
            run_case(plan, interaction, seed, output, device)
        except Exception:
            failed = True
            write_json(
                Path(output) / f"{interaction}_seed{seed}" / "result.json",
                dict(
                    status="failed",
                    interaction=interaction,
                    seed=seed,
                    traceback=traceback.format_exc(),
                ),
            )
            traceback.print_exc()
        finally:
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
    return 1 if failed else 0


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
    processes, logs = [], []
    for index in range(args.workers):
        env = os.environ.copy()
        env.update(
            CUDA_VISIBLE_DEVICES=str(index) if not args.allow_cpu else "",
            OMP_NUM_THREADS=str(plan["cpu_threads_per_worker"]),
            PYTHONUNBUFFERED="1",
        )
        log = (args.output / f"worker_{index}.log").open("w", encoding="utf-8")
        logs.append(log)
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
        processes.append(
            subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        )
    try:
        while any(p.poll() is None for p in processes):
            summary = summarize(plan, args.output)
            print(
                "Campaign:",
                [(c["interaction"], c["seed"], c["status"]) for c in summary["cases"]],
                flush=True,
            )
            time.sleep(30)
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
                process.wait()
        for log in logs:
            log.close()
    summary = summarize(plan, args.output)
    return (
        0
        if summary["all_completed"] and all(p.returncode == 0 for p in processes)
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
