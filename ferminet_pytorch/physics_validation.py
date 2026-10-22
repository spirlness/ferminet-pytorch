"""Reproducible, time-bounded physical evidence; no precision-based stopping.

Run: python -m ferminet_pytorch.physics_validation --device cuda --budget-seconds 1500
Every checkpoint and full production trajectory is retained for independent audit.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

import torch

from .model import FermiNet
from .hamiltonian import MolecularSystem, compute_local_energy
from .evaluation import evaluate_frozen_wavefunction
from .systems import build_h2_system
from .vmc_driver import build_cosine_scheduler
from .vmc_driver import VMCDriver, TrainingConfig
from .mcmc import MCMCSampler


def provenance(root):
    files = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.glob("*.py"))
    }

    def git(*args):
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=root,
                capture_output=True,
                text=True,
                encoding="utf-8",
                check=False,
            )
        except FileNotFoundError:
            return None
        return result.stdout if result.returncode == 0 else None

    commit, diff = git("rev-parse", "HEAD"), git("diff", "HEAD")
    return dict(
        commit=commit.strip() if commit else None,
        diff_sha256=hashlib.sha256(diff.encode()).hexdigest()
        if diff is not None
        else None,
        status=git("status", "--short"),
        source_sha256=files,
        python=platform.python_version(),
        torch=torch.__version__,
        cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None,
    )


def write_json(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def verify_output(output):
    """Replay saved checkpoints on CPU against the recorded source snapshot."""
    output = output.resolve()
    report = json.loads((output / "results.json").read_text(encoding="utf-8"))
    if "composable_model.py" in report["environment"]["source_sha256"]:
        raise ValueError(
            "Legacy checkpoints require an explicit offline conversion; runtime compatibility is not provided"
        )
    source = output / "source_snapshot"
    for name, expected in report["environment"]["source_sha256"].items():
        if hashlib.sha256((source / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Source snapshot hash mismatch: {name}")
    spec = importlib.util.spec_from_file_location(
        "_physics_evidence_snapshot",
        source / "__init__.py",
        submodule_search_locations=[str(source)],
    )
    package = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = package
    spec.loader.exec_module(package)
    verification = {"source_snapshot_matches": True, "checkpoints": []}
    for case in report["cases"]:
        if case.get("status") != "completed":
            continue
        path = output / case["checkpoint"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != case["checkpoint_sha256"]:
            raise ValueError(f"Checkpoint hash mismatch: {path.name}")
        data = torch.load(path, map_location="cpu", weights_only=True)
        classes = (FermiNet, package.FermiNet)
        models = [
            cls(
                data["nuclear_coords"],
                data["nuclear_charges"],
                data["n_up"],
                data["n_down"],
                **data["config"],
            ).double()
            for cls in classes
        ]
        for model in models:
            model.load_state_dict(data["state_dict"])
        torch.manual_seed(119)
        points = (
            torch.randn(8, data["n_up"] + data["n_down"], 3, dtype=torch.float64) + 0.2
        )
        points[0, 1] = points[0, 0] + 1e-4
        values = []
        for model, system_class, energy_fn in [
            (models[0], MolecularSystem, compute_local_energy),
            (models[1], package.MolecularSystem, package.compute_local_energy),
        ]:
            system = system_class(
                data["nuclear_coords"],
                data["nuclear_charges"],
                data["n_up"],
                data["n_down"],
            )
            log, sign = model.forward_log_psi(points)
            energy = energy_fn(model, system, points.clone().requires_grad_())[0]
            values.append((log.detach(), sign.detach(), energy.detach()))
        for actual, expected in zip(*values):
            torch.testing.assert_close(actual, expected, atol=1e-9, rtol=1e-9)
        verification["checkpoints"].append(
            dict(
                name=case["name"],
                passed=True,
                max_log_difference=float((values[0][0] - values[1][0]).abs().max()),
                max_energy_difference=float((values[0][2] - values[1][2]).abs().max()),
            )
        )
    verification["final_source_sha256"] = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(Path(__file__).resolve().parent.glob("*.py"))
    }
    write_json(
        output
        / (
            "verification_"
            + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            + ".json"
        ),
        verification,
    )
    print(
        "Verified source snapshot and checkpoints:",
        len(verification["checkpoints"]),
        flush=True,
    )
    return verification


def run_case(case, output, device, deadline):
    name, architecture, seed, species, train_limit, eval_limit = case
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    if species == "He":
        system = MolecularSystem(
            torch.zeros(1, 3, device=device, dtype=torch.float64),
            torch.tensor([2.0], device=device, dtype=torch.float64),
            1,
            1,
        )
    else:
        system = build_h2_system(
            bond_length=3.0 if species == "H2_stretched" else 1.4,
            device=device,
            dtype=torch.float64,
        )
    cls = FermiNet
    config = dict(
        num_determinants=4,
        hidden_dim_one=32,
        hidden_dim_two=8,
        num_layers=2,
        use_cusp_jastrow=True,
        spin_adapted=True,
        envelope_type="gaussian",
    )
    if architecture == "psiformer":
        config["interaction_type"] = "psiformer"
    model = cls(system.nuclear_coords, system.nuclear_charges, 1, 1, **config).to(
        device=device, dtype=torch.float64
    )
    sampler = MCMCSampler(model, system, num_walkers=256, step_size=0.5)
    sampler.burn_in(50)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    driver = VMCDriver(
        model,
        system,
        sampler,
        optimizer,
        scheduler=build_cosine_scheduler(optimizer, 600, min_ratio=0.01),
        config=TrainingConfig(num_epochs=600, mcmc_steps_per_epoch=10),
    )
    start = time.perf_counter()
    for epoch in range(600):
        if time.perf_counter() >= min(deadline, start + train_limit):
            break
        metrics = driver.step()
        if epoch % 100 == 0:
            print(
                name, "epoch", epoch + 1, "energy", metrics["energy_mean"], flush=True
            )
    train_seconds = time.perf_counter() - start
    checkpoint = output / (name + ".pt")
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": config,
            "seed": seed,
            "nuclear_coords": system.nuclear_coords.cpu(),
            "nuclear_charges": system.nuclear_charges.cpu(),
            "n_up": 1,
            "n_down": 1,
        },
        checkpoint,
    )
    history_path = output / (name + "_training.json")
    write_json(history_path, driver.history)
    result = dict(
        name=name,
        architecture=architecture,
        species=species,
        seed=seed,
        config=config,
        dtype="float64",
        walkers=256,
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        checkpoint=checkpoint.name,
        epochs=len(driver.history),
        train_seconds=train_seconds,
        training_stop="epochs" if len(driver.history) == 600 else "time_budget",
        training_history=history_path.name,
        learning_rate=1e-3,
        scheduler_epochs=600,
        burn_in_steps=50,
        selection="final_checkpoint_no_energy_selection",
    )
    remaining = deadline - time.perf_counter()
    if remaining > 5:
        result["evaluation"] = evaluate_frozen_wavefunction(
            model,
            system,
            num_walkers=256,
            num_steps=512,
            burn_in_steps=50,
            decorrelation_steps=10,
            device=device,
            dtype=torch.float64,
            seed=seed + 10000,
            verbose=False,
            max_seconds=min(eval_limit, max(1.0, remaining - 5)),
            trajectory_path=str(output / (name + "_trajectory.pt")),
        )
    else:
        result["evaluation"] = {"diagnostics_status": "not_run_budget_exhausted"}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--budget-seconds", type=float, default=1500.0)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--verify-output",
        type=Path,
        help="CPU replay against an existing source_snapshot and checkpoints",
    )
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.verify_output is not None:
        verify_output(args.verify_output)
        return
    if args.budget_seconds <= 0 or args.budget_seconds > 1800:
        parser.error("budget must be in (0, 1800] seconds")
    device = torch.device(args.device)
    root = Path(__file__).resolve().parent
    output = args.output or Path.cwd() / "physics_evidence" / datetime.now(
        timezone.utc
    ).strftime("%Y%m%dT%H%M%SZ")
    output.mkdir(parents=True, exist_ok=False)
    snapshot = output / "source_snapshot"
    snapshot.mkdir()
    for path in [*root.glob("*.py"), root.parent / "pyproject.toml"]:
        (snapshot / path.name).write_bytes(path.read_bytes())
    report = dict(
        environment=provenance(root),
        budget_seconds=args.budget_seconds,
        started_utc=datetime.now(timezone.utc).isoformat(),
        cases=[],
        seed_summary={},
    )
    cases = [
        (f"{arch}_H2_seed{seed}", arch, seed, "H2", 100.0, 70.0)
        for arch in ("ferminet", "psiformer")
        for seed in (11, 22, 33)
    ]
    cases += [
        ("ferminet_He_seed11", "ferminet", 11, "He", 60.0, 40.0),
        ("ferminet_stretched_seed11", "ferminet", 11, "H2_stretched", 60.0, 40.0),
    ]
    start = time.perf_counter()
    deadline = start + args.budget_seconds
    for case in cases:
        if time.perf_counter() >= deadline - 10:
            report["cases"].append(
                {"name": case[0], "status": "not_run_budget_exhausted"}
            )
            continue
        try:
            item = run_case(case, output, device, deadline)
            item["status"] = "completed"
        except Exception:
            item = {
                "name": case[0],
                "status": "failed",
                "error": traceback.format_exc(),
            }
        report["cases"].append(item)
        report["elapsed_seconds"] = time.perf_counter() - start
        write_json(output / "results.json", report)
        print(case[0], item.get("evaluation", item.get("error")), flush=True)
    for architecture in ("ferminet", "psiformer"):
        runs = [
            v
            for v in report["cases"]
            if v.get("architecture") == architecture
            and v.get("species") == "H2"
            and "energy_mean" in v.get("evaluation", {})
        ]
        if len(runs) == 3:
            means = torch.tensor(
                [v["evaluation"]["energy_mean"] for v in runs], dtype=torch.float64
            )
            within = (
                math.sqrt(sum(v["evaluation"]["energy_sem"] ** 2 for v in runs)) / 3
            )
            between = float(means.std(unbiased=True)) / math.sqrt(3)
            # Three independent training runs, df=2 Student t critical value.
            sem = max(within, between)
            half = 4.3026527299 * sem
            report["seed_summary"][architecture] = dict(
                mean_of_checkpoint_energies=float(means.mean()),
                sem=max(within, between),
                ci95=[float(means.mean()) - half, float(means.mean()) + half],
                method="t_df2_max_within_between_SEM",
                all_diagnostics_passed=all(
                    v["evaluation"]["diagnostics_status"] == "passed" for v in runs
                ),
            )
    report["elapsed_seconds"] = time.perf_counter() - start
    write_json(output / "results.json", report)
    print("Evidence saved:", output, flush=True)


if __name__ == "__main__":
    main()
