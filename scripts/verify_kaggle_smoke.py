"""Validate downloaded real two-T4, one-epoch artifacts before a long launch."""

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import torch


def plan_digest(plan):
    return hashlib.sha256(
        json.dumps(plan, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def verify(download, kernel, expected_revision=None, continuation=False):
    root = Path(__file__).resolve().parents[1]
    download = Path(download)
    base = download / "ferminet_results"
    environment = json.loads((base / "bootstrap.json").read_text())
    summary = json.loads((base / "summary.json").read_text())
    small = json.loads(
        (
            root
            / (
                "configs/kaggle_continue_smoke.json"
                if continuation
                else "configs/kaggle_smoke.json"
            )
        ).read_text()
    )
    large = json.loads(
        (
            root
            / (
                "configs/kaggle_continue_t4x2.json"
                if continuation
                else "configs/kaggle_t4x2.json"
            )
        ).read_text()
    )
    origin = small["continuation"]["source_epochs"] if continuation else 0
    revision = environment["source_commit"]
    if expected_revision is not None and revision != expected_revision:
        raise ValueError("Smoke source revision does not match")
    if len(environment["gpu_names"]) != 2 or any(
        "T4" not in name for name in environment["gpu_names"]
    ):
        raise ValueError("Smoke must actually run on two T4 GPUs")
    if not summary["all_completed"] or len(summary["cases"]) != 2:
        raise ValueError("Both smoke cases must complete")
    if (
        small["training"]["num_epochs"] != origin + 1
        or small["model"] != large["model"]
        or small["training"]["num_walkers"] != large["training"]["num_walkers"]
    ):
        raise ValueError("Smoke must use one epoch and the long model/batch dimensions")
    if {c["interaction"] for c in summary["cases"]} != set(large["interactions"]):
        raise ValueError("Both long-run backbones must be checked")
    if {c["device"]["visible_devices"] for c in summary["cases"]} != {"0", "1"}:
        raise ValueError("Smoke cases must use distinct physical GPUs")
    for case in summary["cases"]:
        directory = base / f"{case['interaction']}_seed{case['seed']}"
        manifest = json.loads((directory / "manifest.json").read_text())
        if manifest["commit"] != revision or manifest["plan"] != small:
            raise ValueError(
                "Smoke manifest or configuration differs from the tested source"
            )
        if (
            case["epochs"] != origin + 1
            or not case["precision_before"]["passed"]
            or not case["precision_after"]["passed"]
        ):
            raise ValueError("One epoch and both numerical audits must pass")
        checkpoint = directory / "final_checkpoint.pt"
        if (
            hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            != case["checkpoint_sha256"]
        ):
            raise ValueError("Final checkpoint hash mismatch")
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if saved["epoch"] != origin + 1 or len(saved["history_rows"]) != origin + 1:
            raise ValueError("Checkpoint does not contain exactly one update")
        if continuation:
            parent_hash = large["continuation"]["checkpoint_sha256"][
                f"{case['interaction']}_seed{case['seed']}"
            ]
            if (
                case.get("updates_this_campaign") != 1
                or saved.get("continuation", {}).get("parent_checkpoint_sha256")
                != parent_hash
                or case.get("continuation", {}).get("parent_epoch") != origin
                or saved["scheduler_state_dict"]["last_epoch"] != 1
            ):
                raise ValueError(
                    "Continuation smoke did not preserve the parent/schedule origin"
                )
            performance = case.get("sampling_performance")
            if not performance or performance.get("selected") not in (
                "eager",
                "compiled",
            ):
                raise ValueError("Sampling execution was not measured")
            if performance["selected"] == "compiled" and not performance.get(
                "signs_equal"
            ):
                raise ValueError("Compiled sampling numerical comparison is missing")
        trajectory = torch.load(
            directory / "evaluation_trajectory.pt", weights_only=True
        )
        if trajectory.dtype != torch.float64 or tuple(trajectory.shape) != (
            small["evaluation"]["num_steps"],
            small["evaluation"]["num_walkers"],
        ):
            raise ValueError("Independent FP64 trajectory is missing or incomplete")
        if not torch.isfinite(trajectory).all():
            raise ValueError("Nonfinite frozen evaluation observation")
    for index in range(2):
        log = (base / f"worker_{index}.log").read_text()
        if (
            '"event": "training_step"' not in log
            or '"event": "case_completed"' not in log
        ):
            raise ValueError(
                "Worker did not emit observable training/completion events"
            )
        runtime = (base / f"runtime_tests_gpu{index}.log").read_text()
        count = re.search(r"Ran (\d+) tests", runtime)
        if (
            not count
            or int(count[1]) < 3
            or "\nOK" not in runtime
            or "skipped=" in runtime
        ):
            raise ValueError("Each GPU must pass the runtime tests without skips")
    console = (download / f"{kernel.split('/')[1]}.log").read_text()
    try:
        records = json.loads(console)
        if isinstance(records, list):
            console = "\n".join(
                record.get("data", "") for record in records if isinstance(record, dict)
            )
    except json.JSONDecodeError:
        pass
    for index in range(2):
        if not any(
            f"[worker {index}]" in line and '"event": "training_step"' in line
            for line in console.splitlines()
        ):
            raise ValueError(
                "Kaggle console did not receive both workers' live training events"
            )
    return dict(
        passed=True,
        checked_utc=datetime.now(timezone.utc).isoformat(),
        smoke_kernel=kernel,
        source_commit=revision,
        epochs_per_case=1,
        parent_epoch=origin,
        continuation=continuation,
        real_gpu_count=2,
        console_forwarding_verified=True,
        long_plan_sha256=plan_digest(large),
        summary_sha256=hashlib.sha256((base / "summary.json").read_bytes()).hexdigest(),
        scope="Execution/numerical/logging gate; not chemical accuracy evidence",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", type=Path, required=True)
    parser.add_argument("--kernel", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--continuation", action="store_true")
    args = parser.parse_args()
    receipt = verify(args.download, args.kernel, args.revision, args.continuation)
    args.receipt.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
