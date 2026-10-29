"""Measure whole VMC updates with/without batched production acceptance counts."""

import argparse
import json
import time
from pathlib import Path
from types import MethodType

import torch

from ferminet_pytorch import FermiNet, TrainingConfig, VMCDriver
from ferminet_pytorch.loss import VMCLoss
from ferminet_pytorch.mcmc import MCMCSampler
from ferminet_pytorch.systems import build_h2_system

from .sampling_performance import configure_sampling


def step_loop(self, num_steps=10, adapt=False):
    previous = self.adapt_step_size
    self.adapt_step_size = adapt
    try:
        for _ in range(num_steps):
            self.step()
    finally:
        self.adapt_step_size = previous
    return self.walkers.clone().detach()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--compiled", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; no CPU fallback")
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    plan = json.loads(
        (
            Path(__file__).resolve().parents[1] / "configs/kaggle_continue_t4x2.json"
        ).read_text()
    )
    report = dict(
        gpu=torch.cuda.get_device_name(),
        torch=str(torch.__version__),
        cuda=torch.version.cuda,
        measured_steps=args.steps,
        warmup_steps=10,
        results=[],
    )
    for interaction in plan["interactions"]:
        timings = {}
        for mode in (
            ("batched_counts", "compiled")
            if args.compiled
            else ("step_loop", "batched_counts")
        ):
            system = build_h2_system(device="cuda", dtype=torch.float32)
            model = FermiNet(
                system.nuclear_coords,
                system.nuclear_charges,
                1,
                1,
                **plan["model"],
                use_cusp_jastrow=True,
                spin_adapted=True,
                interaction_type=interaction,
            ).cuda()
            config = TrainingConfig(
                **plan["training"], device="cuda", seed=11, verbose=False
            )
            sampler = MCMCSampler(model, system, num_walkers=512)
            driver = VMCDriver(
                model,
                system,
                sampler,
                config=config,
                loss_fn=VMCLoss(
                    config.clip_factor, config.clip_method, max_invalid_fraction=0
                ),
            )
            path = args.parent / f"{interaction}_seed11/final_checkpoint.pt"
            driver.extend_from_checkpoint(
                path,
                expected_sha256=plan["continuation"]["checkpoint_sha256"][
                    f"{interaction}_seed11"
                ],
            )
            if mode == "step_loop":
                sampler.sample = MethodType(step_loop, sampler)
            if mode == "compiled":
                performance = configure_sampling(sampler, 30000, mode="compiled")
                print(json.dumps(performance), flush=True)
            for _ in range(10):
                driver.step()
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(args.steps):
                driver.step()
            torch.cuda.synchronize()
            timings[mode] = (time.perf_counter() - start) / args.steps
            print(
                json.dumps(
                    dict(
                        interaction=interaction,
                        mode=mode,
                        seconds_per_epoch=timings[mode],
                    )
                ),
                flush=True,
            )
            del driver, sampler, model
        report["results"].append(
            dict(
                interaction=interaction,
                **timings,
                speedup=timings["batched_counts"] / timings["compiled"]
                if args.compiled
                else timings["step_loop"] / timings["batched_counts"],
            )
        )
    report["scope"] = (
        "Same checkpoint/configuration, complete exact-kinetic VMC updates; step_loop is the current reference, not a different physical estimator"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
