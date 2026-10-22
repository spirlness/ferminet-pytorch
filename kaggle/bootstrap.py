"""Rendered by scripts.prepare_kaggle with an immutable public Git commit."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    repository = globals().get("SOURCE_REPOSITORY")
    revision = globals().get("SOURCE_COMMIT")
    config = globals().get("CAMPAIGN_CONFIG")
    if not repository or not revision or not config:
        raise RuntimeError(
            "Render this bootstrap with python -m scripts.prepare_kaggle"
        )
    root = Path("/kaggle/working")
    source = root / "ferminet_source"
    output = root / "ferminet_results"
    output.mkdir(parents=True, exist_ok=True)
    if not source.exists():
        subprocess.run(["git", "init", "-q", str(source)], check=True)
        subprocess.run(
            ["git", "-C", str(source), "remote", "add", "origin", repository],
            check=True,
        )
    subprocess.run(
        ["git", "-C", str(source), "fetch", "--depth", "1", "origin", revision],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(source), "checkout", "--detach", "FETCH_HEAD"], check=True
    )
    resolved = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if resolved != revision:
        raise RuntimeError(
            "Fetched source commit does not match the submitted revision"
        )
    import torch
    import numpy as np

    if globals().get("RESUME_KERNEL"):
        candidates = [
            p.parent
            for p in Path("/kaggle/input").rglob("campaign.json")
            if p.parent.name == "ferminet_results"
            and json.loads(p.read_text()).get("commit") == revision
        ]
        if len(candidates) != 1:
            raise RuntimeError(
                "Resume requires exactly one mounted campaign with the same commit"
            )
        shutil.copytree(candidates[0], output, dirs_exist_ok=True)

    names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    environment = dict(
        source_repository=repository,
        source_commit=resolved,
        python=sys.version,
        torch=str(torch.__version__),
        numpy=np.__version__,
        cuda=torch.version.cuda,
        gpu_names=names,
        config=config,
        resume_kernel=globals().get("RESUME_KERNEL"),
    )
    (output / "bootstrap.json").write_text(
        json.dumps(environment, indent=2), encoding="utf-8"
    )
    print(json.dumps(environment, indent=2), flush=True)
    if len(names) != 2 or any("T4" not in name for name in names):
        raise RuntimeError("This submission requires exactly two real T4 GPUs")
    # Keep Kaggle's CUDA-enabled torch; no large wheel replacement or dependency
    # resolver. Module execution from the pinned checkout supplies the package.
    env = os.environ.copy()
    env.update(PYTHONUNBUFFERED="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    for device in range(2):
        test_env = dict(env, CUDA_VISIBLE_DEVICES=str(device))
        test_path = output / f"runtime_tests_gpu{device}.log"
        with test_path.open("w", encoding="utf-8") as log:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "unittest",
                    "tests.test_campaign_runtime.TestCampaignCUDA",
                    "-v",
                ],
                cwd=source,
                env=test_env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
    subprocess.run(
        [
            sys.executable,
            "-u",
            "-m",
            "scripts.kaggle_campaign",
            "--config",
            str(source / config),
            "--output",
            str(output),
            "--workers",
            "2",
        ],
        cwd=source,
        env=env,
        check=True,
    )


if __name__ == "__main__":
    main()
