"""Choose a sampling-only compiled forward by measured amortized benefit.

Coordinate derivatives and parameter-gradient forwards remain eager and FP32.
No Metropolis transitions or random draws are performed during this selection.
"""

import time
from pathlib import Path

import torch

from ferminet_pytorch import VMCDriver


def configure_sampling(sampler, remaining_updates, mode="auto", mcmc_steps=10):
    eager = sampler.model.forward_log_psi
    sampler.forward_log_psi_fn = eager
    report = dict(
        requested=mode,
        selected="eager",
        scope="Sampling forward only; exact kinetic remains eager",
    )
    if mode == "eager" or sampler.device.type != "cuda":
        return report
    if mode not in ("auto", "compiled"):
        raise ValueError("Unknown sampling execution mode")
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state(sampler.device)
    compile_start = time.perf_counter()
    try:
        compiled = torch.compile(eager, fullgraph=True)
        coordinates = sampler.walkers.detach()
        with torch.no_grad():
            compiled(coordinates)
            torch.cuda.synchronize(sampler.device)
            compile_seconds = time.perf_counter() - compile_start
            differences = []
            for delta in (0.0, 0.01, -0.02):
                probe = coordinates + delta
                expected_log, expected_sign = eager(probe)
                actual_log, actual_sign = compiled(probe)
                torch.testing.assert_close(
                    actual_log, expected_log, atol=1e-4, rtol=1e-6
                )
                if not torch.equal(actual_sign, expected_sign):
                    raise FloatingPointError("Compiled sampling signs differ")
                finite = torch.isfinite(expected_log)
                differences.append(
                    float((actual_log[finite] - expected_log[finite]).abs().max())
                )

            def measure(function):
                for _ in range(3):
                    function(coordinates)
                torch.cuda.synchronize(sampler.device)
                started = time.perf_counter()
                for _ in range(30):
                    function(coordinates)
                torch.cuda.synchronize(sampler.device)
                return (time.perf_counter() - started) / 30

            eager_seconds = measure(eager)
            compiled_seconds = measure(compiled)
            saving = (
                (eager_seconds - compiled_seconds)
                * remaining_updates
                * (mcmc_steps + 1)
            )
        report.update(
            compile_seconds=compile_seconds,
            eager_seconds=eager_seconds,
            compiled_seconds=compiled_seconds,
            estimated_seconds_saved=saving,
            max_log_difference=max(differences),
            signs_equal=True,
        )
        if mode == "compiled" or (
            compiled_seconds < 0.95 * eager_seconds and saving > compile_seconds
        ):
            sampler.forward_log_psi_fn = compiled
            report["selected"] = "compiled"
        sampler.recompute_log_psi()
    except Exception as error:
        sampler.forward_log_psi_fn = eager
        sampler.recompute_log_psi()
        if mode == "compiled":
            raise
        report.update(
            reason=type(error).__name__,
            compile_seconds=time.perf_counter() - compile_start,
        )
    finally:
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state(cuda_rng, sampler.device)
    return report


def configure_driver_sampling(driver, directory, remaining_updates, mode="auto"):
    """Benchmark complete updates, restoring all training and random state.

    A faster forward alone can still slow complete VMC updates. ABBA blocks
    reduce order bias; their optimizer updates never enter the real trajectory.
    """
    report = configure_sampling(
        driver.sampler,
        remaining_updates,
        mode,
        mcmc_steps=driver.config.mcmc_steps_per_epoch,
    )
    if report["selected"] != "compiled":
        return driver, report
    sampler = driver.sampler
    compiled = sampler.forward_log_psi_fn
    eager = driver.model.forward_log_psi
    path = Path(directory) / "sampling_probe.pt"
    driver.save_checkpoint(path)

    def restore(function):
        sampler.forward_log_psi_fn = function
        restored = VMCDriver(
            driver.model,
            driver.system,
            sampler,
            config=driver.config,
            loss_fn=driver.loss_fn,
            kinetic_mode=driver.kinetic_mode,
            num_probes=driver.num_probes,
        )
        restored.load_checkpoint(path)
        return restored

    timings = {"eager": [], "compiled": []}
    try:
        for name in ("compiled", "eager", "eager", "compiled"):
            trial = restore(compiled if name == "compiled" else eager)
            for _ in range(3):
                trial.step()
            torch.cuda.synchronize(sampler.device)
            started = time.perf_counter()
            for _ in range(8):
                trial.step()
            torch.cuda.synchronize(sampler.device)
            timings[name].append((time.perf_counter() - started) / 8)
        baseline = sum(timings["eager"]) / 2
        candidate = sum(timings["compiled"]) / 2
        saving = (baseline - candidate) * remaining_updates
        selected = (
            "compiled"
            if mode == "compiled"
            or (candidate < 0.95 * baseline and saving > report["compile_seconds"])
            else "eager"
        )
        report.update(
            selected=selected,
            vmc_eager_seconds=baseline,
            vmc_compiled_seconds=candidate,
            vmc_estimated_seconds_saved=saving,
            probe_updates_discarded=44,
            complete_training_state_restored=True,
        )
        restored = restore(compiled if selected == "compiled" else eager)
    except Exception:
        restore(eager)
        raise
    finally:
        path.unlink()
    return restored, report
