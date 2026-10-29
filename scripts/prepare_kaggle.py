"""Render a private Kaggle submission that fetches this exact public commit."""

import argparse
import json
import re
import subprocess
from pathlib import Path

from .verify_kaggle_smoke import plan_digest


def prepare(owner, mode, output, revision=None, resume_kernel=None, gate_path=None):
    root = Path(__file__).resolve().parents[1]
    gate = None
    continuing = mode.startswith("continue-")
    smoke = mode in ("smoke", "continue-smoke")
    config = (
        (
            "configs/kaggle_continue_smoke.json"
            if smoke
            else "configs/kaggle_continue_t4x2.json"
        )
        if continuing
        else ("configs/kaggle_smoke.json" if smoke else "configs/kaggle_t4x2.json")
    )
    large_config = (
        "configs/kaggle_continue_t4x2.json"
        if continuing
        else "configs/kaggle_t4x2.json"
    )
    if not smoke:
        if gate_path is None:
            raise ValueError(
                "A verified one-epoch smoke gate is required before a long submission"
            )
        gate = json.loads(Path(gate_path).read_text())
        revision = revision or gate.get("source_commit")
        large = json.loads((root / large_config).read_text())
        if not (
            gate.get("passed") is True
            and gate.get("epochs_per_case") == 1
            and gate.get("real_gpu_count") == 2
            and gate.get("console_forwarding_verified") is True
            and gate.get("source_commit") == revision
            and gate.get("long_plan_sha256") == plan_digest(large)
        ):
            raise ValueError("Smoke gate does not match this source/configuration")
    revision = (
        revision
        or subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip()
    )
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Use a full immutable Git commit SHA")
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", owner):
        raise ValueError("Invalid Kaggle owner")
    title = (
        (
            "FermiNet T4x2 Continue Smoke"
            if smoke
            else "FermiNet T4x2 Continue Multiseed"
        )
        if continuing
        else ("FermiNet T4x2 Smoke" if smoke else "FermiNet T4x2 Multiseed")
    )
    slug = title.lower().replace(" ", "-")
    preamble = (
        'SOURCE_REPOSITORY = "https://github.com/spirlness/ferminet-pytorch.git"\n'
        f"SOURCE_COMMIT = {revision!r}\nCAMPAIGN_CONFIG = {config!r}\n\n"
    )
    if gate is not None:
        preamble += f"SMOKE_GATE = {gate!r}\n\n"
    parent_kernel = None
    if continuing:
        parent_kernel = json.loads((root / config).read_text())["continuation"][
            "source_kernel"
        ]
        preamble += f"PARENT_KERNEL = {parent_kernel!r}\n\n"
    if resume_kernel:
        if not re.fullmatch(r"[a-zA-Z0-9_-]+/[a-zA-Z0-9_-]+", resume_kernel):
            raise ValueError("Resume kernel must be owner/slug")
        preamble += f"RESUME_KERNEL = {resume_kernel!r}\n\n"
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "run.py").write_text(
        preamble + (root / "kaggle/bootstrap.py").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    metadata = dict(
        id=f"{owner}/{slug}",
        title=title,
        code_file="run.py",
        language="python",
        kernel_type="script",
        is_private=True,
        enable_gpu=True,
        enable_internet=True,
        machine_shape="NvidiaTeslaT4",
        dataset_sources=[],
        competition_sources=[],
        kernel_sources=list(
            dict.fromkeys(x for x in (parent_kernel, resume_kernel) if x)
        ),
    )
    (output / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owner", required=True)
    parser.add_argument(
        "--mode",
        choices=["smoke", "long", "continue-smoke", "continue-long"],
        default="smoke",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--revision", help="Full SHA; preserve the original source on resume"
    )
    parser.add_argument(
        "--resume-kernel", help="Mount prior output and resume matching cases"
    )
    parser.add_argument(
        "--gate", type=Path, help="Verified one-epoch receipt, mandatory for long mode"
    )
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(
                args.owner,
                args.mode,
                args.output,
                args.revision,
                args.resume_kernel,
                args.gate,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
