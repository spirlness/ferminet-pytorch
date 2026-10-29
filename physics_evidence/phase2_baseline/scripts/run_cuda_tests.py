"""Run the production test suite with CUDA required and every skip rejected.

Run from the checkout: python -m scripts.run_cuda_tests
Each invocation saves a new evidence directory, leaving earlier runs intact.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    output = args.output or ROOT / "physics_evidence" / datetime.now(timezone.utc).strftime("phase1_cuda_%Y%m%dT%H%M%SZ")
    output = output.resolve()
    if not args.child:
        output.mkdir(parents=True, exist_ok=False)
        env = os.environ.copy()
        for key in ("CUDA_VISIBLE_DEVICES", "CPATH", "C_INCLUDE_PATH",
                    "CPLUS_INCLUDE_PATH", "INCLUDE", "LIB", "LIBPATH", "LIBRARY_PATH"):
            env.pop(key, None)
        tcc = Path(sys.prefix) / "Lib/site-packages/triton/runtime/tcc/tcc.exe"
        if tcc.is_file():
            env["CC"] = str(tcc)
        env.update(OMP_NUM_THREADS="1", PYTHONUTF8="1")
        with (output / "tests.log").open("w", encoding="utf-8") as log:
            result = subprocess.run(
                [sys.executable, "-m", "scripts.run_cuda_tests", "--child", "--output", str(output)],
                cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
            )
        summary = output / "summary.json"
        print(summary.read_text(encoding="utf-8") if summary.exists()
              else f"No summary produced; inspect {output / 'tests.log'}")
        return result.returncode

    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA must be available; CPU fallback is not allowed")
    torch.set_num_threads(1)
    print("CUDA:", torch.cuda.get_device_name(0), flush=True)
    start = time.perf_counter()
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"), top_level_dir=str(ROOT))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    torch.cuda.synchronize()
    summary = dict(
        tests_run=result.testsRun, failures=len(result.failures), errors=len(result.errors),
        skipped=result.skipped, elapsed_seconds=time.perf_counter()-start,
        torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0),
        no_skip_requirement_met=not result.skipped,
        successful=result.wasSuccessful() and not result.skipped and result.testsRun > 0,
        source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                       for directory in ("ferminet_pytorch", "tests", "scripts")
                       for p in sorted((ROOT / directory).glob("*.py"))},
    )
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("NO_SKIP_SUMMARY", json.dumps({k:v for k,v in summary.items() if k != "source_sha256"}), flush=True)
    return 0 if summary["successful"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
