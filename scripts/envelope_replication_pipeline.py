"""One idempotent monitoring/dispatch tick for the authorized replication."""

import argparse
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

from .envelope_replication import batches, read_plan
from .envelope_replication_evidence import collect, gate, prepare, verify_batch
from .kaggle_campaign import write_json
from .verify_kaggle_smoke import plan_digest


def tick(args):
    plan = read_plan()
    args.output.mkdir(parents=True, exist_ok=True)
    journal = args.output / "pipeline.json"
    identity = {"source_commit": args.revision, "protocol_sha256": plan_digest(plan)}
    state = (
        json.loads(journal.read_text())
        if journal.exists()
        else dict(**identity, phase="new", downloads=[])
    )
    if any(state.get(k) != v for k, v in identity.items()):
        raise ValueError("Pipeline identity differs")

    def record(phase, **values):
        state.update(phase=phase, **values)
        write_json(journal, state)
        return state

    def command(*arguments):
        result = subprocess.run(
            [args.kaggle, *arguments],
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
        )
        if result.returncode:
            raise RuntimeError(result.stdout + result.stderr)
        return result.stdout

    if state["phase"] in ("new", "batch_ready"):
        smoke = state["phase"] == "new"
        index = state.get("next_batch", 0)
        target = args.output / ("smoke" if smoke else f"batch_{index}")
        metadata = prepare(
            SimpleNamespace(
                revision=args.revision,
                gate=None if smoke else args.output / "gate.json",
                smoke=smoke,
                batch_index=index,
                output=target,
                owner=args.owner,
                resume_kernel=None,
            ),
            plan,
        )
        # Persist intent BEFORE the external mutation; an ambiguous submission is never blindly retried.
        record(
            "submission_pending",
            submitting_kind="smoke" if smoke else "batch",
            submitting_batch=index,
            active_kernel=metadata["id"],
            submission_directory=str(target.resolve()),
        )
        raw = command(
            "kernels",
            "push",
            "-p",
            str(target.resolve()),
            "--accelerator",
            "NvidiaTeslaT4",
            "-t",
            "2400" if smoke else "32400",
        )
        (target / "submission.txt").write_text(raw, encoding="utf-8")
        return record(
            "smoke_submitted" if smoke else "batch_submitted", active_batch=index
        )
    if state["phase"] == "submission_pending":
        raise RuntimeError(
            "Submission outcome requires read-only remote/source reconciliation; refusing duplicate push"
        )
    if state["phase"] in ("smoke_submitted", "batch_submitted"):
        status = command("kernels", "status", state["active_kernel"])
        if "COMPLETE" not in status and not any(
            s in status for s in ("ERROR", "FAILED", "CANCEL")
        ):
            return dict(state, remote_status=status.strip())
        smoke = state["phase"] == "smoke_submitted"
        directory = args.download_root / (
            "smoke" if smoke else f"batch_{state['active_batch']}"
        )
        directory.mkdir(parents=True, exist_ok=True)
        # A failed/incomplete transfer leaves the submitted phase intact, so the next tick retries collection.
        command(
            "kernels",
            "output",
            state["active_kernel"],
            "-p",
            str(directory.resolve()),
            "--page-size",
            "100",
            "--quiet",
        )
        if "COMPLETE" not in status:
            write_json(
                directory / "remote_failure.json",
                {"kernel": state["active_kernel"], "status": status},
            )
            return record(
                "remote_failed",
                failed_output=str(directory.resolve()),
                remote_status=status.strip(),
            )
        if smoke:
            receipt = gate(directory, args.revision, plan, args.history.parent)
            write_json(args.output / "gate.json", receipt)
            return record(
                "batch_ready" if receipt["passed"] else "budget_rejected",
                seeds_per_batch=receipt["seeds_per_batch"],
                next_batch=0,
            )
        verify_batch(directory, args.revision, plan)
        downloaded = list(state["downloads"])
        if str(directory.resolve()) not in downloaded:
            downloaded.append(str(directory.resolve()))
        following = state["active_batch"] + 1
        if following < len(batches(plan, state["seeds_per_batch"])):
            return record("batch_ready", downloads=downloaded, next_batch=following)
        return record("all_downloaded", downloads=downloaded)
    if state["phase"] == "all_downloaded":
        if not args.archive.exists():
            collect(
                [Path(p) for p in state["downloads"]],
                args.archive,
                args.revision,
                plan,
                args.history,
            )
        else:
            summary = json.loads((args.archive / "verified_summary.json").read_text())
            if (
                summary["source_commit"] != args.revision
                or summary["primary"]["status"] == "incomplete"
            ):
                raise ValueError("Existing archive is incomplete or differs")
        return record("complete", report=str((args.archive / "REPORT.md").resolve()))
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--kaggle", required=True)
    parser.add_argument("--owner", default="hwbwhzbsh")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--download-root", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--history", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(tick(args), indent=2), flush=True)


if __name__ == "__main__":
    main()
