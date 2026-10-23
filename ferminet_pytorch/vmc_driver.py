"""One VMC optimization core shared by CLI, experiments and direct use."""

import copy
import hashlib
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import torch
from torch import nn

from .kinetic_modern import KineticMode
from .loss import VMCLoss
from .mcmc import MCMCSampler
from .model import FermiNet
from .systems import build_h2_system


@dataclass(frozen=True)
class TrainingConfig:
    num_walkers: int = 512
    num_epochs: int = 50
    mcmc_steps_per_epoch: int = 10
    burn_in_steps: int = 50
    lr: float = 0.001
    step_size: float = 0.5
    target_acceptance: Optional[float] = None
    move_mode: str = "all"
    clip_factor: float = 5.0
    clip_method: str = "tanh"
    grad_clip: float = 1.0
    use_cusp_jastrow: bool = False
    two_stage_jastrow: bool = False
    jastrow_stage1_epochs: Optional[int] = None
    jastrow_lr: Optional[float] = None
    eval_after_train: bool = False
    eval_steps: int = 50
    eval_decorrelation_steps: int = 10
    spin_adapted: bool = False
    envelope_type: Optional[str] = None
    smooth_cuspless: Optional[bool] = None
    restore_best: bool = True
    checkpoint_path: Optional[str] = None
    interaction_type: str = "ferminet"
    compile_sampler: bool = False
    lr_schedule: str = "cosine"
    lr_warmup_epochs: int = 500
    lr_decay_epochs: int = 10000
    lr_min_ratio: float = 0.1
    device: Optional[str] = None
    dtype: torch.dtype = torch.float32
    seed: int = 42
    verbose: bool = True

    def __post_init__(self):
        if self.lr_schedule not in ("cosine", "warmup_inverse"):
            raise ValueError("Unknown learning-rate schedule")
        if self.lr_warmup_epochs < 0 or self.lr_decay_epochs < 1:
            raise ValueError("Invalid learning-rate warmup/decay length")
        if not 0 < self.lr_min_ratio <= 1:
            raise ValueError("Learning-rate minimum ratio must lie in (0, 1]")
        if (
            self.num_walkers < 1
            or self.num_epochs < 0
            or self.burn_in_steps < 0
            or self.mcmc_steps_per_epoch < 1
        ):
            raise ValueError(
                "Walker count and sampling steps must be positive; epochs/burn-in nonnegative"
            )
        if (
            self.jastrow_stage1_epochs is not None
            and not 0 <= self.jastrow_stage1_epochs <= self.num_epochs
        ):
            raise ValueError("Jastrow warmup must lie within the training schedule")
        if self.lr <= 0 or self.jastrow_lr is not None and self.jastrow_lr <= 0:
            raise ValueError("Learning rates must be positive")


def build_cosine_scheduler(optimizer, total_steps, min_ratio=0.01):
    def factor(step):
        fraction = min(max(step, 0) / max(total_steps, 1), 1.0)
        return min_ratio + 0.5 * (1 - min_ratio) * (1 + math.cos(math.pi * fraction))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def build_training_scheduler(optimizer, total_steps, config):
    if config.lr_schedule == "cosine":
        return build_cosine_scheduler(optimizer, total_steps)

    def factor(step):
        if step < config.lr_warmup_epochs:
            return config.lr_min_ratio + (1 - config.lr_min_ratio) * (
                step / config.lr_warmup_epochs
            )
        decay = (step - config.lr_warmup_epochs) / config.lr_decay_epochs
        return max(config.lr_min_ratio, 1.0 / (1.0 + decay))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


class VMCDriver:
    """Own optimization state; the sampler alone owns walkers and their cache."""

    def __init__(
        self,
        model,
        system,
        sampler,
        optimizer=None,
        scheduler=None,
        config=None,
        loss_fn=None,
        kinetic_mode=KineticMode.REVERSE,
        num_probes=1,
    ):
        if sampler.model is not model or sampler.system is not system:
            raise ValueError("Driver and sampler must share the same model and system")
        self.model, self.system, self.sampler = model, system, sampler
        self.config = config or TrainingConfig()
        self.loss_fn = loss_fn or VMCLoss(
            self.config.clip_factor, self.config.clip_method
        )
        self.kinetic_mode, self.num_probes = kinetic_mode, num_probes
        self.epoch, self.stage = 0, 1
        self.history, self.best = [], {}
        self.continuation = None
        self.is_two_stage = (
            self.config.two_stage_jastrow and model.has_trainable_jastrow()
        )
        self.stage1_epochs = (
            (
                self.config.jastrow_stage1_epochs
                if self.config.jastrow_stage1_epochs is not None
                else max(1, self.config.num_epochs // 2)
            )
            if self.is_two_stage
            else self.config.num_epochs
        )
        self.stage1_epochs = min(self.stage1_epochs, self.config.num_epochs)
        if self.is_two_stage:
            model.freeze_jastrow()
        self.optimizer = optimizer or torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad], lr=self.config.lr
        )
        self.scheduler = scheduler or build_training_scheduler(
            self.optimizer,
            self.stage1_epochs if self.is_two_stage else self.config.num_epochs,
            self.config,
        )

    def restore(self, state):
        self.model.load_state_dict(state)
        self.sampler.recompute_log_psi()

    def _advance_stage(self):
        if self.is_two_stage and self.stage == 1 and self.epoch >= self.stage1_epochs:
            if self.config.restore_best and 1 in self.best:
                self.restore(self.best[1][2])
            self._configure_stage_two()
            self.stage = 2

    def _configure_stage_two(self):
        self.model.unfreeze_jastrow()
        groups = self.model.get_parameter_groups(
            self.config.lr,
            self.config.jastrow_lr
            if self.config.jastrow_lr is not None
            else self.config.lr * 0.1,
        )
        self.optimizer = torch.optim.Adam(groups)
        self.scheduler = build_training_scheduler(
            self.optimizer, self.config.num_epochs - self.stage1_epochs, self.config
        )

    def step(self):
        start = time.perf_counter()
        self._advance_stage()
        walkers = self.sampler.sample(self.config.mcmc_steps_per_epoch, adapt=False)
        loss, stats, _ = self.loss_fn.compute_step_loss(
            self.model,
            self.system,
            walkers,
            kinetic_mode=self.kinetic_mode,
            num_probes=self.num_probes,
        )
        energy = float(stats["energy_mean"])
        # Energy belongs to theta_t: snapshot BEFORE optimizer.step().
        if math.isfinite(energy):
            for key in (0, self.stage):
                if key not in self.best or energy < self.best[key][0]:
                    self.best[key] = (
                        energy,
                        self.epoch + 1,
                        copy.deepcopy(self.model.state_dict()),
                    )
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.config.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
        self.optimizer.step()
        self.sampler.recompute_log_psi()
        self.scheduler.step()
        self.epoch += 1
        metrics = dict(
            stats,
            epoch_time=time.perf_counter() - start,
            stage=self.stage,
            mcmc_acceptance=self.sampler.last_acceptance_rate,
            mcmc_step_size=self.sampler.step_size,
        )
        self.history.append(metrics)
        return metrics

    @property
    def training_history(self):
        fields = dict(
            energy_raw="energy_mean",
            energy_clipped="energy_clipped_mean",
            energy_var="energy_var",
            loss="loss",
            acceptance_rate="mcmc_acceptance",
            step_size="mcmc_step_size",
            num_valid="num_valid",
            num_invalid="num_invalid",
            stage="stage",
        )
        return {
            name: [row[key] for row in self.history] for name, key in fields.items()
        }

    @property
    def selected_best(self):
        key = 2 if self.is_two_stage and 2 in self.best else 0
        return (
            self.best[key]
            if key in self.best
            else (float("inf"), 0, copy.deepcopy(self.model.state_dict()))
        )

    def save_checkpoint(self, path):
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        energy, epoch, state = self.selected_best
        torch.save(
            dict(
                schema_version=2,
                config=asdict(self.config),
                loss_settings={
                    name: getattr(self.loss_fn, name, None)
                    for name in (
                        "clip_factor",
                        "clip_method",
                        "max_invalid_fraction",
                    )
                },
                kinetic_mode=str(KineticMode(self.kinetic_mode).value),
                num_probes=self.num_probes,
                optimizer_class=type(self.optimizer).__qualname__,
                scheduler_class=type(self.scheduler).__qualname__,
                epoch=self.epoch,
                best_epoch=epoch,
                best_energy=energy,
                model_state_dict=self.model.state_dict(),
                best_model_state_dict=state,
                optimizer_state_dict=self.optimizer.state_dict(),
                scheduler_state_dict=self.scheduler.state_dict(),
                history=self.training_history,
                history_rows=self.history,
                best_records=self.best,
                walkers=self.sampler.walkers,
                step_size=self.sampler.step_size,
                rng_state=torch.get_rng_state(),
                cuda_rng_state=torch.cuda.get_rng_state(self.sampler.device)
                if self.sampler.device.type == "cuda"
                else None,
                sampler_settings={
                    name: getattr(self.sampler, name)
                    for name in (
                        "target_acceptance",
                        "adapt_step_size",
                        "adapt_rate",
                        "move_mode",
                        "last_acceptance_rate",
                        "total_steps",
                        "accepted_steps",
                        "proposed_moves",
                        "accepted_moves",
                    )
                },
                stage=self.stage,
                continuation=self.continuation,
            ),
            target.with_name(target.name + ".tmp"),
        )
        os.replace(target.with_name(target.name + ".tmp"), target)

    def load_checkpoint(self, path):
        """Resume schema-2 Adam/cosine runs on the same dtype/device backend.

        Exact RNG continuation is tested within one software/hardware environment;
        this is not a cross-version or cross-hardware reproducibility guarantee.
        """
        saved = torch.load(path, map_location="cpu", weights_only=True)
        return self._restore_checkpoint(saved)

    def extend_from_checkpoint(self, path, *, expected_sha256):
        """Explicit schedule migration; preserve Adam, walkers/RNG and lineage.

        This starts a new experiment from a verified completed stage-2 state.
        It is distinct from exact same-configuration checkpoint resumption.
        """
        digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        if digest != expected_sha256:
            raise ValueError("Parent checkpoint SHA-256 mismatch")
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if (
            saved.get("stage") != 2
            or saved.get("epoch") != saved["config"]["num_epochs"]
            or saved.get("continuation") is not None
            or self.config.num_epochs <= saved["epoch"]
            or self.config.lr_schedule != "warmup_inverse"
        ):
            raise ValueError(
                "Extension requires a completed stage-2 parent and a longer tail schedule"
            )
        self._restore_checkpoint(saved, extending=True)
        self.continuation = dict(
            parent_checkpoint_sha256=digest,
            parent_epoch=saved["epoch"],
            parent_config=saved["config"],
            optimizer_state_preserved=True,
            schedule="warmup_inverse",
        )
        for group in self.optimizer.param_groups:
            peak = (
                (
                    self.config.jastrow_lr
                    if self.config.jastrow_lr is not None
                    else self.config.lr * 0.1
                )
                if group.get("name") == "jastrow"
                else self.config.lr
            )
            group["lr"] = group["initial_lr"] = peak
        self.scheduler = build_training_scheduler(
            self.optimizer, self.config.num_epochs - self.epoch, self.config
        )
        return self

    def _restore_checkpoint(self, saved, extending=False):
        if self.epoch or self.history:
            raise ValueError("Load a checkpoint into a freshly constructed driver")
        if saved.get("schema_version") != 2:
            raise ValueError("Resume requires checkpoint schema version 2")
        ignored = {"device", "checkpoint_path", "verbose"}
        if extending:
            ignored |= {
                "num_epochs",
                "lr",
                "jastrow_lr",
                "lr_schedule",
                "lr_warmup_epochs",
                "lr_decay_epochs",
                "lr_min_ratio",
            }
        current = {k: v for k, v in asdict(self.config).items() if k not in ignored}
        prior = {
            k: v
            for k, v in asdict(TrainingConfig(**saved["config"])).items()
            if k not in ignored
        }
        if current != prior:
            raise ValueError("Checkpoint training configuration does not match")
        loss_settings = {
            name: getattr(self.loss_fn, name, None) for name in saved["loss_settings"]
        }
        if (
            loss_settings != saved["loss_settings"]
            or saved["kinetic_mode"] != KineticMode(self.kinetic_mode).value
            or saved["num_probes"] != self.num_probes
        ):
            raise ValueError("Checkpoint loss/kinetic configuration does not match")
        if saved["optimizer_class"] != "Adam" or saved["scheduler_class"] != "LambdaLR":
            raise ValueError("Resume currently supports the shared Adam/cosine driver")
        if (
            type(self.optimizer) is not torch.optim.Adam
            or type(self.scheduler) is not torch.optim.lr_scheduler.LambdaLR
        ):
            raise ValueError("Resume requires the default Adam/cosine driver")
        walkers = saved["walkers"]
        if (
            walkers.shape != self.sampler.walkers.shape
            or walkers.dtype != self.sampler.dtype
        ):
            raise ValueError("Checkpoint walker shape/dtype does not match")
        if (saved["cuda_rng_state"] is not None) != (
            self.sampler.device.type == "cuda"
        ):
            raise ValueError("Resume must use the original CPU/CUDA backend")
        self.model.load_state_dict(saved["model_state_dict"])
        self.epoch, self.stage = saved["epoch"], saved["stage"]
        if self.stage == 2 and self.is_two_stage:
            self._configure_stage_two()
        self.optimizer.load_state_dict(saved["optimizer_state_dict"])
        self.scheduler.load_state_dict(saved["scheduler_state_dict"])
        self.history, self.best = saved["history_rows"], saved["best_records"]
        self.continuation = saved.get("continuation")
        self.sampler.walkers = walkers.to(self.sampler.device)
        self.sampler.step_size = saved["step_size"]
        for name, value in saved["sampler_settings"].items():
            setattr(self.sampler, name, value)
        self.sampler.recompute_log_psi()
        torch.set_rng_state(saved["rng_state"])
        if saved["cuda_rng_state"] is not None:
            torch.cuda.set_rng_state(saved["cuda_rng_state"], self.sampler.device)
        return self

    def run(self, num_epochs=None, callback=None):
        count = (
            max(0, self.config.num_epochs - self.epoch)
            if num_epochs is None
            else num_epochs
        )
        for _ in range(count):
            row = self.step()
            if self.config.checkpoint_path is not None:
                self.save_checkpoint(self.config.checkpoint_path)
            if callback is not None:
                callback(self.epoch, row)
        return self.history

    def result(self, eval_result=None):
        energy, epoch, state = self.selected_best
        last = self.history[-1] if self.history else {}
        result = dict(
            model=self.model,
            system=self.system,
            sampler=self.sampler,
            history=self.training_history,
            final_energy=last.get("energy_mean", float("nan")),
            final_variance=last.get("energy_var", float("nan")),
            best_model_state=state,
            best_energy=energy,
            best_epoch=epoch,
            is_two_stage=self.is_two_stage,
            stage1_epochs=self.stage1_epochs if self.is_two_stage else None,
            eval_result=eval_result,
        )
        for number in (1, 2):
            record = self.best.get(number, (float("inf"), 0, None))
            result[f"stage{number}_best_energy"] = (
                record[0] if self.is_two_stage else None
            )
            result[f"stage{number}_best_epoch"] = (
                record[1] if self.is_two_stage else None
            )
        return result


def train_vmc(system=None, model=None, config=None):
    """Set up a run, execute the sole driver, then optionally evaluate frozen Psi."""
    config = config or TrainingConfig()
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    device = torch.device(
        config.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    system = (
        build_h2_system(device=device, dtype=config.dtype)
        if system is None
        else system.to(device=device, dtype=config.dtype)
    )
    if model is None:
        model = FermiNet(
            system.nuclear_coords,
            system.nuclear_charges,
            system.n_up,
            system.n_down,
            num_determinants=8,
            use_cusp_jastrow=config.use_cusp_jastrow,
            spin_adapted=config.spin_adapted,
            envelope_type=config.envelope_type,
            smooth_cuspless=config.smooth_cuspless,
            interaction_type=config.interaction_type,
        )
    model = model.to(device=device, dtype=config.dtype)
    sampler = MCMCSampler(
        model,
        system,
        num_walkers=config.num_walkers,
        step_size=config.step_size,
        target_acceptance=config.target_acceptance,
        move_mode=config.move_mode,
        device=device,
        dtype=config.dtype,
        forward_log_psi_fn=torch.compile(model.forward_log_psi)
        if config.compile_sampler
        else None,
    )
    driver = VMCDriver(model, system, sampler, config=config)
    sampler.burn_in(config.burn_in_steps)

    def log(epoch, row):
        if config.verbose:
            print(
                f"Epoch {epoch}: E={row['energy_mean']:.8f} Ha; Var={row['energy_var']:.5f}; "
                f"acceptance={row['mcmc_acceptance']:.3f}; stage={row['stage']}",
                flush=True,
            )

    driver.run(callback=log)
    if config.restore_best and driver.best:
        driver.restore(driver.selected_best[2])
    evaluation = None
    if config.eval_after_train:
        from .evaluation import evaluate_frozen_wavefunction

        evaluation = evaluate_frozen_wavefunction(
            model,
            system,
            num_walkers=config.num_walkers,
            num_steps=config.eval_steps,
            decorrelation_steps=config.eval_decorrelation_steps,
            device=device,
            dtype=config.dtype,
            verbose=config.verbose,
        )
    return driver.result(evaluation)
