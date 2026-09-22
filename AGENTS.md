# Repository Guidelines

## Project Structure & Module Organization

This repository is a flat Python package. Core neural-network components live in `features.py`, `layers.py`, `envelope.py`, and `model.py`; physics and optimization code is in `hamiltonian.py`, `mcmc.py`, `mcmc_stats.py`, `loss.py`, and `train_vmc.py`; plotting and diagnostics live in `visualize.py`. Runnable examples and performance checks are `example_h2.py`, `benchmark_compile.py`, and `benchmark_100_atoms.py`.

Tests remain beside the source as `test_*.py`:

- `test_ferminet.py` — core physics properties and network architecture
- `test_vmc.py` — MCMC sampling, gradient estimation, and energy clipping
- `test_phase2.py` — kinetic-energy solvers, nuclear-charge injection, and the anisotropic envelope
- `test_visualize.py` — visualization module
- `test_mcmc_stats.py` — chain autocorrelation statistics, checked against analytic AR(1) results
- `test_physics_invariants.py` — physical invariants and the limitations that are knowingly not met (symmetry, cusp, parameter reachability, estimator behaviour)
- `test_geometry_invariants.py` — geometry-feature invariants and numerical stability
- `test_estimator_invariants.py` — energy-estimator and potential numerical invariants
- `test_cli_guards.py` — CLI guards that keep random estimators off the reported-energy path
- `test_robustness_guards.py` — boundary behaviours and guards: determinant-weight recovery, non-finite handling, solver fallback visibility, geometry consistency

Documentation is in `README.md` and `BEGINNER_GUIDE.md`. Reviewed audit reports, their reproduction scripts, and their measured JSON outputs live in `audit_review/`; `audit_review/README.md` indexes them and records the current fix status. Checked-in figures belong under `assets/` or `assets/plots/`.

## Build, Test, and Development Commands

- `uv sync` creates or updates the environment from `pyproject.toml` and `uv.lock`.
- `uv run python example_h2.py` runs the small H2 inference and local-energy example.
- `uv run python train_vmc.py --epochs 5 --walkers 64` performs a short local training smoke test.
- `uv run python -m unittest discover -s . -p "test_*.py" -v` runs the complete test suite.
- `uv run python -m unittest test_vmc.py` runs one focused test module.
- `uv build` produces source and wheel distributions through Hatchling.

GPU-heavy benchmarks are optional; use `uv run python benchmark_100_atoms.py --double --quick` for the reduced path.

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

This affects only the `torch.compile` path. With the pinned environment the suite is green either way; on older torch the single test that exercises `compile_model()` skips itself with that reason.

## Coding Style & Naming Conventions

Use Python 3.9+ and four-space indentation. Follow the existing PEP 8-like layout: `snake_case` for functions, variables, and modules; `PascalCase` for classes; and `UPPER_CASE` for constants. Add type annotations to public APIs and document tensor shapes, dtypes, and physical units where relevant. Preserve the established `(B, N, 3)` electron-coordinate convention. No formatter or linter is configured, so keep imports grouped, lines readable, and changes narrowly scoped.

## Testing Guidelines

Tests use the standard-library `unittest` framework. Name files `test_<area>.py`, classes `Test<Feature>`, and methods `test_<behavior>`. Seed PyTorch for stochastic cases and test physical invariants—shapes, finiteness, antisymmetry, and numerical tolerances—not only successful execution. Add regression coverage with every behavioral fix. The repository does not define a coverage threshold; the full suite must pass before review.

## Commit & Pull Request Guidelines

History follows Conventional Commit-style subjects such as `feat:`, `fix(visualize):`, `docs:`, and `style:`. Keep the subject imperative and describe one logical change. Pull requests should explain the motivation and numerical/physical impact, list commands run, and link related issues. Include updated plots or screenshots when visualization output changes, plus benchmark numbers when making performance claims. Do not commit virtual environments, checkpoints, logs, or generated build artifacts.
