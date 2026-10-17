# Repository Guidelines & AI Agent Conventions

## Project Structure & Module Organization

This repository implements a high-performance, composable FermiNet framework for solving the multi-electron Schrödinger equation in quantum chemistry. The codebase is organized into modular layers supporting both classic baseline workflows and modern decoupled architectures:

- **Geometry & Invariant Cache**:
  - `geometry.py`: Immutable `GeometricContext` for centralized displacement and smooth distance calculations (`smooth_distance`).
  - `features.py`: `ElectronFeatureEmbedding` (single- and pair-electron streams, smooth cuspless distances).

- **Equivariant Interaction Layers**:
  - `layers.py`: Baseline `FermiInteractionLayer` with spin-separated pooling.
  - `layers_modern.py`: `FermiInteractionLayerModern` with zero high-frequency tensor allocations.
  - `psiformer_attention.py`: `PsiformerAttentionBlock` self-attention mechanism without explicit 4D pair tensor storage.

- **Nuclear Envelopes & Physical Factors**:
  - `envelope.py` & `envelope_modern.py`: `NuclearEnvelope` and `NuclearEnvelopeModern` (exponential and smooth gaussian).
  - `jastrow.py`: `KatoCuspJastrow` (Padé dampening satisfying electron-nuclear and electron-electron cusp conditions).
  - `spin.py`: `SpinProjector` (singlet spatial symmetrization $1/2(I + P_{12})$ and multi-electron spin projection polynomials).

- **Deep Wavefunction Models**:
  - `model.py`: Classic unified `FermiNet`.
  - `composable_model.py`: `ComposableFermiNet`, `OrbitalEnvelopeProjector`, and `BranchlessCofactorDeterminant`.
  - `legacy_adapters.py`: Non-breaking adapters (`FermiNetLegacyAdapter`, `...LegacyAdapter`) ensuring 100% backward compatibility.

- **Physics Operators & Modern Kinetic Solvers**:
  - `kinetic_modern.py`: `KineticEnergyCalculator` supporting `REVERSE` (autograd), `HUTCHINSON` (stochastic trace estimation), and `FORWARD_JVP`.
  - `hamiltonian.py`: `MolecularSystem`, Coulomb potential energies, `compute_local_energy`, and `diagnose_cusp_divergence`.

- **Sampling, Loss, & Decoupled VMC Driver**:
  - `mcmc.py`: `MCMCSampler` with vectorized all-electron Gaussian walks and acceptance tuning.
  - `loss.py`: `VMCLoss`, `clip_local_energy` (Umrigar tanh clipping), and `vmc_surrogate_loss`.
  - `vmc_driver.py`: Decoupled four-pillar driver (`VariationalState`, `HamiltonianComposite`, `MetropolisSampler`, `VMCDriver`).
  - `train_vmc.py`: CLI training entrypoint, cosine schedulers, and two-stage warmup utilities.

Documentation:
- `README.md` — Framework overview, dual-track quickstart, and feature matrix.
- `BEGINNER_GUIDE.md` — Zero-to-one conceptual guide and intuitive physics explanation.
- `DEVELOPER_GUIDE.md` — Modern architecture, extension patterns, and deep-dive technical tutorials.
- `ARCHITECTURE_REDUNDANCY_REVIEW_AND_REFACTORING_PROPOSAL.md` — Systematic architectural audit and design rationale.
- `FERMINET_REFACTORING_EVALUATION_AND_PRODUCTION_READINESS_REPORT.md` — Level-5 production-readiness verification and benchmarks.

---

## Build, Test, and Development Commands

- `uv sync` creates or updates the environment from `pyproject.toml` and `uv.lock`.
- `uv run python example_h2.py` runs the H2 forward inference and local-energy decomposition.
- `uv run python train_vmc.py --epochs 5 --walkers 64` runs a quick CPU end-to-end VMC test.
- `uv run python -m unittest discover -s . -p "test_*.py" -v` executes the complete unit test suite on CPU (CUDA tests skipped).
- `uv run python -m unittest test_refactor_review_regressions.py -v` executes the modern composable architecture regression suite.
- `uv build` builds source and wheel distributions using Hatchling.

### CUDA / GPU Development with Conda `torch` Environment

For GPU-accelerated training and CUDA numerical regression testing (RTX 3060+ / CUDA 12.x), use the pre-configured Conda `torch` environment (`F:\conda\envs\torch` or `conda activate torch`):

- `F:\conda\envs\torch\python.exe train_vmc.py --device cuda --epochs 5 --walkers 128` runs GPU end-to-end VMC training.
- `F:\conda\envs\torch\python.exe -m unittest test_cuda.py -v` executes the 18 GPU-specific numerical regression tests.
- `F:\conda\envs\torch\python.exe -m unittest discover -s . -p "test_*.py" -v` executes the complete 246-test suite with GPU acceleration enabled (all tests passing, 0 skipped).

### Windows / Non-UTF-8 Locales and `torch.compile`

If importing the inductor backend fails with `UnicodeDecodeError: 'gbk' codec can't decode byte 0x94`, set `PYTHONUTF8=1` in the shell before launching python:
```powershell
$env:PYTHONUTF8='1'; python -m unittest test_compile.py
```

---

## Coding Standards & Architectural Contracts

1. **Single-Forward Contract for VMC**:
   - Compute local energy $E_L(\mathbf{r}) = T_L(\mathbf{r}) + V(\mathbf{r})$ on coordinate gradients with `create_graph=False`. Always `.detach()` the local energy before computing surrogate loss $\mathcal{L}(\theta) = 2 \mathbb{E}[(E_{\text{clip}} - \bar{E}_{\text{clip}}) \ln|\Psi_\theta|]$. Never retain higher-order graph connections between coordinate Laplacians and model parameter updates.
2. **Zero-Allocation and Zero Graph Breaks**:
   - In forward passes, avoid in-place tensor mutations, dynamic scalar indexing on device tensors, or dynamic shapes that trigger TorchDynamo graph breaks.
   - Use vectorized batch linear algebra (`torch.linalg.slogdet`, batched einsum/matmul).
3. **Preserve Coordinate Conventions**:
   - Standard electron coordinate shape is strictly `(B, N, 3)` where $B$ is the batch size (walkers), $N$ is total electrons ($n_\uparrow + n_\downarrow$).
   - Nuclear coordinates are `(M, 3)` and nuclear charges are `(M,)`.
4. **Physical Invariants Verification**:
   - Any modification to wavefunction models must preserve:
     1. Antisymmetry under permutation of same-spin electrons;
     2. Singlet total spin projection: $\hat{S}^2 \Psi_{\text{singlet}} = 0$;
     3. Kato cusp conditions: $\partial \ln|\Psi| / \partial r_{iI} \to -Z_I$ and $\partial \ln|\Psi| / \partial r_{ij} \to 1/2$ or $1/4$;
     4. Energy local finiteness at coalescence points ($E_L < \infty$).

---

## Commit & Pull Request Guidelines

Follow Conventional Commits: `feat:`, `fix:`, `refactor:`, `docs:`, `test:`, or `perf:`. Keep PRs scoped, explain physical and performance impacts, and ensure all tests pass (`uv run python -m unittest discover -s . -p "test_*.py"`).
