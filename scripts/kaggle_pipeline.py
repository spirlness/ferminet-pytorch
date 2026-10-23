"""Submit a pinned smoke run, verify its artifacts, then optionally submit long training.

Runs in the CUDA Python environment; --kaggle points to the authenticated CLI.
The journal makes re-running the command resume its last successful stage.
"""

import argparse
import json
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from .prepare_kaggle import prepare
from .verify_kaggle_smoke import plan_digest, verify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--continuation", action="store_true")
    parser.add_argument("--submit-long", action="store_true")
    parser.add_argument("--kaggle", default=shutil.which("kaggle"))
    parser.add_argument("--output", type=Path, default=Path(".kaggle_deploy/pipeline"))
    parser.add_argument("--timeout-seconds", type=int, default=2400)
    args = parser.parse_args()
    if not args.kaggle:
        parser.error("The authenticated Kaggle CLI is required")
    root = Path(__file__).resolve().parents[1]
    plan = json.loads(
        (
            root
            / (
                "configs/kaggle_continue_t4x2.json"
                if args.continuation
                else "configs/kaggle_t4x2.json"
            )
        ).read_text()
    )
    directory = args.output.resolve() / args.revision[:12]
    directory.mkdir(parents=True, exist_ok=True)
    journal = directory / "pipeline.json"
    identity = dict(
        source_commit=args.revision,
        long_plan_sha256=plan_digest(plan),
        continuation=args.continuation,
    )
    state = (
        json.loads(journal.read_text())
        if journal.exists()
        else dict(**identity, phase="new")
    )
    if any(state.get(k) != v for k, v in identity.items()):
        raise ValueError("Pipeline journal belongs to a different source/configuration")

    def record(phase, **fields):
        state.update(
            phase=phase, updated_utc=datetime.now(timezone.utc).isoformat(), **fields
        )
        temporary = journal.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(state, indent=2), encoding="utf-8")
        temporary.replace(journal)
        print(json.dumps(state), flush=True)

    def command(*arguments):
        result = subprocess.run(
            [args.kaggle, *arguments],
            cwd=root,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        if result.returncode:
            raise RuntimeError(result.stdout + result.stderr)
        return result.stdout

    prefix = "continue-" if args.continuation else ""
    small = directory / "smoke"
    metadata = prepare(args.owner, prefix + "smoke", small, revision=args.revision)
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
                "1200",
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
            if any(token in status for token in ("ERROR", "CANCEL", "FAILED")):
                raise RuntimeError("Smoke did not complete successfully: " + status)
            if time.monotonic() - start >= args.timeout_seconds:
                raise TimeoutError(
                    "Smoke still pending/running; re-run this command to resume polling"
                )
            time.sleep(15)
        download = directory / "smoke_output"
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
        receipt = verify(download, metadata["id"], args.revision, args.continuation)
        receipt_path = directory / "gate.json"
        receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
        record("smoke_passed", receipt=str(receipt_path))
    if args.submit_long and state["phase"] == "smoke_passed":
        large = directory / "long"
        metadata = prepare(
            args.owner,
            prefix + "long",
            large,
            revision=args.revision,
            gate_path=state["receipt"],
        )
        print(
            command(
                "kernels",
                "push",
                "-p",
                str(large),
                "--accelerator",
                "NvidiaTeslaT4",
                "-t",
                "36000",
            ),
            flush=True,
        )
        record("long_submitted", long_kernel=metadata["id"])
    print(json.dumps(state, indent=2), flush=True)


if __name__ == "__main__":
    main()
