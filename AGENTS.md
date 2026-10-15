# Repository Guidelines

## Project Structure & Module Organization

This repository is a flat Python package. Core neural-network and physics components live in `features.py`, `layers.py`, `envelope.py`, `jastrow.py`, `spin.py`, and `model.py`; physics and optimization code is in `hamiltonian.py`, `mcmc.py`, `loss.py`, and `train_vmc.py`. Runnable tutorial examples are `example_h2.py` and `train_vmc.py`.

Tests remain beside the source as `test_*.py`:

- `test_ferminet.py` — core physics properties and network architecture
- `test_vmc.py` — MCMC sampling, gradient estimation, and energy clipping
- `test_spin_and_cusp.py` — spin-adapted projection, Kato cusp derivative limits, and singularity cancellation

Documentation is in `README.md` and `BEGINNER_GUIDE.md`. Optional architecture diagrams live under `assets/`.

## Build, Test, and Development Commands

- `uv sync` creates or updates the environment from `pyproject.toml` and `uv.lock`.
- `uv run python example_h2.py` runs the small H2 inference and local-energy example.
- `uv run python train_vmc.py --epochs 5 --walkers 64` performs a short local training smoke test (CPU default).
- `uv run python -m unittest discover -s . -p "test_*.py" -v` runs the test suite on CPU (CUDA tests skipped).
- `uv run python -m unittest test_vmc.py` runs one focused test module.
- `uv build` produces source and wheel distributions through Hatchling.

### CUDA / GPU Development with Conda `torch` Environment

For GPU-accelerated training and CUDA numerical regression testing (RTX 3060+), use the pre-configured local Conda `torch` environment (`F:\conda\envs\torch` or `conda activate torch`), which bundles CUDA-enabled PyTorch (CUDA 12.x+):

- `F:\conda\envs\torch\python.exe train_vmc.py --device cuda --epochs 5 --walkers 128` runs GPU end-to-end VMC training.
- `F:\conda\envs\torch\python.exe -m unittest test_cuda.py -v` executes the 18 GPU-specific numerical regression tests.
- `F:\conda\envs\torch\python.exe -m unittest discover -s . -p "test_*.py" -v` executes the complete 145-test suite with GPU acceleration enabled (all tests passing, 0 skipped).




**Windows / non-UTF-8 locales and `torch.compile`:** if importing the inductor backend fails with `UnicodeDecodeError: 'gbk' codec can't decode byte 0x94`, your torch predates the fix that makes inductor read its CuteDSL templates with `encoding="utf-8"`. Torch 2.14.0 (the version pinned in `uv.lock`) has that fix; 2.13.0 does not. Setting `PYTHONUTF8=1` works around it on older torch — the variable only takes effect before the interpreter starts, so it cannot be set from code:

```bash
PYTHONUTF8=1 python -m unittest test_ferminet.py   # bash / Git Bash
```
```powershell
$env:PYTHONUTF8='1'; python -m unittest test_ferminet.py   # PowerShell
```
```cmd
set PYTHONUTF8=1 && python -m unittest test_ferminet.py   :: cmd.exe
```
This affects only the optional `torch.compile` Inductor backend path under Windows / non-UTF-8 locales.

## Coding Style & Naming Conventions

Use Python 3.9+ and four-space indentation. Follow the existing PEP 8-like layout: `snake_case` for functions, variables, and modules; `PascalCase` for classes; and `UPPER_CASE` for constants. Add type annotations to public APIs and document tensor shapes, dtypes, and physical units where relevant. Preserve the established `(B, N, 3)` electron-coordinate convention. No formatter or linter is configured, so keep imports grouped, lines readable, and changes narrowly scoped.

## Testing Guidelines

Tests use the standard-library `unittest` framework. Name files `test_<area>.py`, classes `Test<Feature>`, and methods `test_<behavior>`. Seed PyTorch for stochastic cases and test physical invariants—shapes, finiteness, antisymmetry, and numerical tolerances—not only successful execution. Add regression coverage with every behavioral fix. The repository does not define a coverage threshold; the full suite must pass before review.

## Commit & Pull Request Guidelines

History follows Conventional Commit-style subjects such as `feat:`, `fix(mcmc):`, `docs:`, and `style:`. Keep the subject imperative and describe one logical change. Pull requests should explain the motivation and numerical/physical impact, list commands run, and link related issues. Do not commit virtual environments, checkpoints, logs, or generated build artifacts.
