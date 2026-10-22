# Repository Guidelines & AI Agent Conventions

## Project Structure & Module Organization

Runtime code lives in `ferminet_pytorch/`, tests in `tests/`, runnable diagnostics in `scripts/`, and demos in `examples/`. Use package imports and module entry points; do not restore flat-module shims or legacy adapters.

This repository implements a FermiNet research framework for solving the multi-electron Schrödinger equation in quantum chemistry. Version 0.3.0 uses one model assembly and training core with FermiNet or Psiformer interactions:

- **Geometry & Invariant Cache**:
  - `geometry.py`: Frozen `GeometricContext` for centralized displacement and smooth distance calculations (`smooth_distance`); tensor contents must be treated as read-only, and caches must not cross coordinate graphs.
  - `features.py`: `ElectronFeatureEmbedding` (single- and pair-electron streams, smooth cuspless distances).

- **Equivariant Interaction Layers**:
  - `layers.py`: Sole `FermiInteractionLayer` with static spin partitions and self-excluded pooling.
  - `psiformer_attention.py`: `PsiformerAttentionBlock` replaces the persistent two-electron hidden stream with attention; pair geometry and attention intermediates still exist.

- **Nuclear Envelopes & Physical Factors**:
  - `envelope.py`: the sole `NuclearEnvelope`, accepting distances `(B,N,M)` and returning `(B,K,N,O)`; exponential and Gaussian variants share this implementation.
  - `jastrow.py`: `KatoCuspJastrow` (Padé dampening satisfying electron-nuclear and electron-electron cusp conditions).
  - `spin.py`: `SpinProjector` (singlet spatial symmetrization $1/2(I + P_{12})$ and multi-electron spin projection polynomials).

- **Deep Wavefunction Models**:
  - `model.py`: Sole `FermiNet`, with FermiNet or Psiformer interactions and common physical factors.

- **Physics Operators & Modern Kinetic Solvers**:
  - `kinetic_modern.py`: `KineticEnergyCalculator` supporting `REVERSE` (autograd), `HUTCHINSON` (stochastic trace estimation), and `FORWARD_JVP`.
  - `hamiltonian.py`: `MolecularSystem`, Coulomb potential energies, `compute_local_energy`, and `diagnose_cusp_divergence`.

- **Sampling, Loss, & Decoupled VMC Driver**:
  - `mcmc.py`: `MCMCSampler` owns walkers/cache and supports all-electron or sequential Gaussian moves; adaptation is limited to burn-in.
  - `loss.py`: `VMCLoss`, `clip_local_energy` (Umrigar tanh clipping), and `vmc_surrogate_loss`.
  - `vmc_driver.py`: `TrainingConfig`, `VMCDriver`, and `train_vmc`; the only optimizer update core. The sampler alone owns walkers/cache.
  - `train_vmc.py`: CLI argument parsing only; system builders live in `systems.py`, scheduling/stages in `vmc_driver.py`.

Documentation:
- `README.md` — Current package quickstart, unified API, commands and documentation index.
- `BEGINNER_GUIDE.md` — Zero-to-one conceptual guide and intuitive physics explanation.
- `DEVELOPER_GUIDE.md` — Current layer/model/driver contracts and extension boundaries.
- `ARCHITECTURE_REDUNDANCY_REVIEW_AND_REFACTORING_PROPOSAL.md` — Implemented 0.3.0 architecture, breaking migration table and evidence limits.
- `ARCHITECTURE_REFACTORING_PROPOSAL_HISTORY.md` — Unmodified old proposal; not current implementation or certification.
- `FERMINET_REFACTORING_EVALUATION_AND_PRODUCTION_READINESS_REPORT.md` — historical report, not current certification.
- `REFACTOR_PHASE1_REPORT.md` — historical package/envelope refactor.
- `REFACTOR_PHASE2_REPORT.md` — current unified layers, model and training verification.

---

## Build, Test, and Development Commands

- `uv sync` creates or updates the environment from `pyproject.toml` and `uv.lock`.
- `uv run python -m examples.example_h2` runs the H2 forward inference and local-energy decomposition.
- `uv run python -m ferminet_pytorch.train_vmc --device cpu --epochs 5 --walkers 64` runs a quick CPU end-to-end VMC test.
- `uv run python -m unittest discover -s tests -t . -v` executes the complete unit test suite on CPU (CUDA tests skipped).
- `uv run python -m unittest tests.test_refactor_review_regressions tests.test_unified_driver -v` executes architecture and unified training regressions.
- `uv build` builds source and wheel distributions using Hatchling.

### CUDA / GPU Development with Conda `torch` Environment

For GPU-accelerated training and CUDA numerical regression testing, use the pre-configured Conda `torch` environment (`F:\conda\envs\torch` or `conda activate torch`). The saved phase-2 acceptance run used an RTX 3060 Laptop GPU, PyTorch 2.14.0 and CUDA runtime 13.0; this is an observed environment, not the minimum supported version:

- `F:\conda\envs\torch\python.exe -m ferminet_pytorch.train_vmc --device cuda --epochs 5 --walkers 128` runs GPU end-to-end VMC training.
- `F:\conda\envs\torch\python.exe -m unittest tests.test_cuda -v` executes GPU-specific numerical, gradient, spin and envelope checks.
- `F:\conda\envs\torch\python.exe -m unittest discover -s tests -t . -v` executes the complete suite, including CUDA cases when available. Report actual pass/skip/failure counts and compiler availability from the run log.
- `F:\conda\envs\torch\python.exe -m scripts.run_cuda_tests` requires CUDA and treats every skipped test as a failure; saves raw output, counts, environment and source hashes. Compiler environment isolation applies only to its child process.

### Windows / Non-UTF-8 Locales and `torch.compile`

If importing the inductor backend fails with `UnicodeDecodeError: 'gbk' codec can't decode byte 0x94`, set `PYTHONUTF8=1` in the shell before launching python:
```powershell
$env:PYTHONUTF8='1'; python -m unittest tests.test_compile
```

---

## Coding Standards & Architectural Contracts

1. **Separate Coordinate and Parameter Graphs for VMC**:
   - Compute local energy $E_L(\mathbf{r}) = T_L(\mathbf{r}) + V(\mathbf{r})$ on coordinate gradients with `create_graph=False`. Always `.detach()` the local energy before computing surrogate loss $\mathcal{L}(\theta) = 2 \mathbb{E}[(E_{\text{clip}} - \bar{E}_{\text{clip}}) \ln|\Psi_\theta|]$. Never retain higher-order graph connections between coordinate Laplacians and model parameter updates.
2. **Allocation Discipline and Compile Compatibility**:
   - In forward passes, avoid in-place tensor mutations, dynamic scalar indexing on device tensors, or dynamic shapes that trigger TorchDynamo graph breaks.
   - Use vectorized batch linear algebra (`torch.linalg.slogdet`, batched einsum/matmul).
3. **Preserve Coordinate Conventions**:
   - Standard electron coordinate shape is strictly `(B, N, 3)` where $B$ is the batch size (walkers), $N$ is total electrons ($n_\uparrow + n_\downarrow$).
   - Nuclear coordinates are `(M, 3)` and nuclear charges are `(M,)`.
4. **Physical Invariants Verification**:
   - Any modification to wavefunction models must preserve:
     1. Antisymmetry under permutation of same-spin electrons;
     2. Singlet total spin projection: $\hat{S}^2 \Psi_{\text{singlet}} = 0$;
     3. Kato cusp conditions apply to spherical-average nonzero components: the electron-nuclear logarithmic slope is $-Z_I$, and the opposite-spin s-wave slope is $1/2$. The $1/4$ slope belongs to the regular radial coefficient after removing the same-spin p-wave zero, not the complete logarithm;
     4. Verify cancellation of Coulomb divergences in the coalescence limit at nonzero wavefunction components. At exact nodes, $H\Psi/\Psi$ is undefined; do not manufacture finite energies by clipping physical distances.

5. **Evidence and Sampling**:
   - Refresh cached log wavefunctions after every optimizer step or state restore.
   - Freeze proposal adaptation during production sampling. Physical Coulomb distances must not use smoothed feature distances.
   - Frozen evaluation rejects nonfinite energies, retains walker/time structure, and reports mixing and reblocking diagnostics. Chemical accuracy requires a matching reference and an uncertainty interval within the tolerance.

---

## Commit & Pull Request Guidelines

Follow Conventional Commits: `feat:`, `fix:`, `refactor:`, `docs:`, `test:`, or `perf:`. Keep PRs scoped, explain physical and performance impacts, and ensure all tests pass (`uv run python -m unittest discover -s tests -t .`).
