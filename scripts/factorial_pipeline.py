"""Gate the immutable factorial campaign on real dual-T4 evidence and timing."""

import argparse
import json
import re
import shutil
import statistics
import subprocess
import time
from pathlib import Path

import torch

from .factorial_campaign import ARMS, schedule, seed_for
from .kaggle_campaign import file_sha, write_json
from .verify_kaggle_smoke import plan_digest

ROOT = Path(__file__).resolve().parents[1]


def read_plan(smoke):
    return json.loads(
        (
            ROOT
            / "configs"
            / ("factorial_smoke.json" if smoke else "factorial_t4x2.json")
        ).read_text()
    )


def project_seconds(cases, long_plan):
    """No energy input: conservative extrapolation of measured execution only."""
    per_update = max(statistics.mean(c["step_seconds"][5:]) for c in cases)
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
    worker_updates = (
        max(len(q) for q in schedule(long_plan)) * long_plan["training"]["num_epochs"]
    )
    worker_records = (
        6 * long_plan["evaluation"]["num_steps"]
        + 3 * long_plan["zero_evaluation_records"]
    )
    # Six training rethermalizations and nine evaluation burn-ins per GPU.
    measured = (
        worker_updates * per_update
        + worker_records * per_record
        + 6 * rethermalize
        + 9 * burn
    )
    projected = measured * long_plan["timing_safety_factor"] + 900
    return {
        "projected_wall_seconds": projected,
        "seconds_per_update": per_update,
        "seconds_per_record_including_observation_analysis": per_record,
        "evaluation_burn_seconds": burn,
        "training_burn_seconds": rethermalize,
        "safety_factor": long_plan["timing_safety_factor"],
        "startup_reserve_seconds": 900,
        "within_budget": projected <= long_plan["max_projected_seconds"],
    }


def verify(download, revision, kernel, parent_output):
    download = Path(download)
    base = download / "ferminet_results"
    small, large = read_plan(True), read_plan(False)
    environment = json.loads((base / "bootstrap.json").read_text())
    campaign = json.loads((base / "campaign.json").read_text())
    summary = json.loads((base / "summary.json").read_text())
    if (
        environment["source_commit"] != revision
        or campaign["commit"] != revision
        or campaign["plan"] != small
        or len(environment["gpu_names"]) != 2
        or any("T4" not in name for name in environment["gpu_names"])
    ):
        raise ValueError("Smoke needs this source/plan on two real T4 GPUs")
    expected_ids = {c["case_id"] for q in schedule(small) for c in q}
    cases = summary["cases"]
    if (
        not summary["all_completed"]
        or {c["case_id"] for c in cases} != expected_ids
        or len(cases) != 4
    ):
        raise ValueError("All four factorial arms must complete")
    initial = []
    for case in cases:
        directory = base / case["case_id"]
        checkpoint = torch.load(
            directory / "final_checkpoint.pt", map_location="cpu", weights_only=True
        )
        state = torch.load(
            directory / "fork_initial_state.pt", map_location="cpu", weights_only=True
        )
        initial.append(state)
        if (
            case["epochs"] != small["training"]["num_epochs"]
            or checkpoint["epoch"] != case["epochs"]
            or len(checkpoint["history_rows"]) != case["epochs"]
            or case["checkpoint_sha256"] != file_sha(directory / "final_checkpoint.pt")
            or case["parent_checkpoint_sha256"]
            != large["fork"]["checkpoint_sha256"]["11"]
            or not case["precision_after"]["passed"]
            or not case["precision_before"]["passed"]
            or not case["evaluation"]["extreme_precision_audit"]["passed"]
            or checkpoint["experiment_origin"]["optimizer_state_preserved"] is not False
        ):
            raise ValueError(
                "Smoke update, parent, checkpoint or precision evidence differs"
            )
        options = ARMS[case["arm"]]
        if (
            case["model_identity"]["envelope_type"] != options["envelope_type"]
            or checkpoint["config"]["clip_method"] != options["clip_method"]
            or checkpoint["model_identity"]["envelope_scale"] != 2.0
            or checkpoint["sampler_settings"]["adapt_step_size"]
        ):
            raise ValueError("Smoke physical factors or production adaptation differ")
        trajectory = torch.load(directory / "evaluation/energy.pt", weights_only=True)
        if (
            trajectory.shape != (small["evaluation"]["num_steps"], 512)
            or not torch.isfinite(trajectory).all()
        ):
            raise ValueError("Smoke held-out trajectory is incomplete/nonfinite")
        observations = json.loads(
            (directory / "evaluation/observations/manifest.json").read_text()
        )
        if (
            not observations["complete"]
            or observations["time_records"] != len(trajectory)
            or observations["seed"] != seed_for(small, case["seed"], case["arm"])
        ):
            raise ValueError("Smoke observation manifest differs")
        energies = []
        for shard in observations["shards"]:
            target = directory / "evaluation/observations" / shard["file"]
            if file_sha(target) != shard["sha256"]:
                raise ValueError("Observation hash differs")
            payload = torch.load(target, weights_only=True)
            for key in (
                "positions",
                "energy",
                "kinetic",
                "potential",
                "log_abs",
                "sign",
            ):
                if not torch.isfinite(payload[key]).all():
                    raise ValueError("Nonfinite smoke observations")
            torch.testing.assert_close(
                payload["energy"],
                payload["kinetic"] + payload["potential"],
                rtol=0,
                atol=1e-12,
            )
            energies.append(payload["energy"])
        torch.testing.assert_close(torch.cat(energies), trajectory, rtol=0, atol=0)
    parent = Path(parent_output) / "ferminet_seed11/final_checkpoint.pt"
    if file_sha(parent) != large["fork"]["checkpoint_sha256"]["11"]:
        raise ValueError("Local reference parent checkpoint hash differs")
    parent_state = torch.load(parent, weights_only=True, map_location="cpu")
    for state in initial:
        for key, parent_key in (
            ("walkers", "walkers"),
            ("rng", "rng_state"),
            ("cuda_rng", "cuda_rng_state"),
        ):
            torch.testing.assert_close(
                state[key], parent_state[parent_key], rtol=0, atol=0
            )
    for envelope in ("gaussian", "soft_exponential"):
        result = json.loads((base / f"zero_seed11_{envelope}/result.json").read_text())
        if (
            result["time_records"] != small["zero_evaluation_records"]
            or result["num_invalid"]
        ):
            raise ValueError("Zero-step evaluation is incomplete")
    for index in (0, 1):
        log = (base / f"runtime_tests_gpu{index}.log").read_text()
        count = re.search(r"Ran (\d+) tests", log)
        if not count or int(count[1]) < 15 or "\nOK" not in log or "skipped=" in log:
            raise ValueError("Each T4 must pass all runtime regressions without skips")
    console = (download / f"{kernel.split('/')[1]}.log").read_text()
    try:
        records = json.loads(console)
        if isinstance(records, list):
            console = "\n".join(
                r.get("data", "") for r in records if isinstance(r, dict)
            )
    except json.JSONDecodeError:
        pass
    for index, queue in enumerate(schedule(small)):
        if not all(
            any(
                f"[worker {index}]" in line
                and '"event": "case_completed"' in line
                and c["case_id"] in line
                for line in console.splitlines()
            )
            for c in queue
        ):
            raise ValueError("Both physical GPUs must forward case completion logs")
    timing = project_seconds(cases, large)
    return {
        "passed": timing["within_budget"],
        "correctness_passed": True,
        "source_commit": revision,
        "long_plan_sha256": plan_digest(large),
        "smoke_plan_sha256": plan_digest(small),
        "real_gpu_count": 2,
        "arms_verified": list(ARMS),
        "updates_per_arm": small["training"]["num_epochs"],
        "timing": timing,
        "limits": "Short-run timings are projections, not guaranteed runtime; no energy-based selection.",
    }


def prepare(owner, revision, directory, smoke=True, gate=None):
    if not re.fullmatch(r"[0-9a-f]{40}", revision) or not re.fullmatch(
        r"[a-zA-Z0-9_-]+", owner
    ):
        raise ValueError("Use a full immutable commit SHA and a valid owner")
    plan = read_plan(smoke)
    if not smoke and not (
        gate
        and gate.get("passed")
        and gate["source_commit"] == revision
        and gate["long_plan_sha256"] == plan_digest(plan)
    ):
        raise ValueError("Formal experiment needs a matching correctness/timing gate")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    slug = "ferminet-factorial-smoke" if smoke else "ferminet-factorial-accuracy"
    preamble = (
        'SOURCE_REPOSITORY = "https://github.com/spirlness/ferminet-pytorch.git"\n'
        f"SOURCE_COMMIT = {revision!r}\n"
        f"CAMPAIGN_CONFIG = {'configs/factorial_smoke.json' if smoke else 'configs/factorial_t4x2.json'!r}\n"
        'CAMPAIGN_MODULE = "scripts.factorial_campaign"\n'
        f"PARENT_KERNEL = {plan['fork']['source_kernel']!r}\n"
        f"SMOKE_GATE = {gate!r}\n\n"
    )
    (directory / "run.py").write_text(
        preamble + (ROOT / "kaggle/bootstrap.py").read_text(), encoding="utf-8"
    )
    metadata = {
        "id": f"{owner}/{slug}",
        "title": "FermiNet Factorial Smoke" if smoke else "FermiNet Factorial Accuracy",
        "code_file": "run.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [plan["fork"]["source_kernel"]],
    }
    write_json(directory / "kernel-metadata.json", metadata)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owner", default="hwbwhzbsh")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--kaggle", default=shutil.which("kaggle"))
    parser.add_argument("--parent-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(".kaggle_deploy/factorial"))
    parser.add_argument("--submit-long", action="store_true")
    parser.add_argument("--collect-long", action="store_true")
    parser.add_argument(
        "--long-output", type=Path, default=Path("campaign_runs/factorial_final")
    )
    parser.add_argument(
        "--archive",
        type=Path,
        default=Path("physics_evidence/factorial_final_20261025"),
    )
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    args = parser.parse_args()
    if not args.kaggle:
        parser.error("Authenticated Kaggle CLI required")
    root = args.output.resolve() / args.revision[:12]
    root.mkdir(parents=True, exist_ok=True)
    journal = root / "pipeline.json"
    identity = {
        "source_commit": args.revision,
        "long_plan_sha256": plan_digest(read_plan(False)),
    }
    state = (
        json.loads(journal.read_text())
        if journal.exists()
        else dict(**identity, phase="new")
    )
    if any(state.get(k) != v for k, v in identity.items()):
        raise ValueError("Pipeline identity differs")

    def command(*arguments):
        result = subprocess.run(
            [args.kaggle, *arguments],
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
        )
        if result.returncode:
            raise RuntimeError(result.stdout + result.stderr)
        return result.stdout

    def record(phase, **values):
        state.update(phase=phase, **values)
        write_json(journal, state)
        print(json.dumps(state), flush=True)

    small = root / "smoke"
    metadata = prepare(args.owner, args.revision, small)
    if state["phase"] == "new":
        print(
            command(
                "kernels",
                "push",
                "-p",
                str(small),
                "--accelerator",
                "NvidiaTeslaT4",
                "-t",
                "2400",
            ),
            flush=True,
        )
        record("smoke_submitted", smoke_kernel=metadata["id"])
    if state["phase"] == "smoke_submitted":
        start = time.monotonic()
        previous = None
        while True:
            status = command("kernels", "status", metadata["id"])
            if status != previous:
                print(status, flush=True)
                previous = status
            if "COMPLETE" in status:
                break
            if any(t in status for t in ("ERROR", "CANCEL", "FAILED")):
                raise RuntimeError("Smoke failed: " + status)
            if time.monotonic() - start >= args.timeout_seconds:
                raise TimeoutError("Resume this pipeline to continue waiting")
            time.sleep(15)
        download = root / "smoke_output"
        command(
            "kernels",
            "output",
            metadata["id"],
            "-p",
            str(download),
            "--page-size",
            "100",
            "--quiet",
        )
        receipt = verify(download, args.revision, metadata["id"], args.parent_output)
        write_json(root / "gate.json", receipt)
        record(
            "smoke_passed" if receipt["passed"] else "timing_rejected",
            receipt=str(root / "gate.json"),
            timing=receipt["timing"],
        )
    if args.submit_long and state["phase"] == "smoke_passed":
        gate = json.loads(Path(state["receipt"]).read_text())
        metadata = prepare(
            args.owner, args.revision, root / "long", smoke=False, gate=gate
        )
        print(
            command(
                "kernels",
                "push",
                "-p",
                str(root / "long"),
                "--accelerator",
                "NvidiaTeslaT4",
                "-t",
                "32400",
            ),
            flush=True,
        )
        record("long_submitted", long_kernel=metadata["id"])
    if args.collect_long and state["phase"] == "long_submitted":
        started = time.monotonic()
        previous = None
        while True:
            status = command("kernels", "status", state["long_kernel"])
            if status != previous:
                print(status, flush=True)
                previous = status
            if "COMPLETE" in status or any(
                t in status for t in ("ERROR", "CANCEL", "FAILED")
            ):
                command(
                    "kernels",
                    "output",
                    state["long_kernel"],
                    "-p",
                    str(args.long_output),
                    "--page-size",
                    "100",
                    "--quiet",
                )
                from .factorial_report import collect

                report = collect(args.long_output, args.archive, args.revision)
                record(
                    "collected",
                    report=str(report.resolve()),
                    final_remote_status=status.strip(),
                )
                break
            if time.monotonic() - started >= args.timeout_seconds:
                raise TimeoutError(
                    "Long experiment remains active; resume with --collect-long"
                )
            time.sleep(15)
    print(json.dumps(state, indent=2), flush=True)
    return 2 if state["phase"] == "timing_rejected" else 0


if __name__ == "__main__":
    raise SystemExit(main())
