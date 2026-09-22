# Repository Guidelines

## Project Structure & Module Organization

This repository is a flat Python package. Core neural-network components live in `features.py`, `layers.py`, `envelope.py`, and `model.py`; physics and optimization code is in `hamiltonian.py`, `mcmc.py`, `loss.py`, and `train_vmc.py`. Runnable examples and performance checks are `example_h2.py`, `benchmark_compile.py`, and `benchmark_100_atoms.py`. Tests remain beside the source as `test_*.py`. Documentation is in `README.md` and `BEGINNER_GUIDE.md`, while checked-in figures belong under `assets/` or `assets/plots/`.

## Build, Test, and Development Commands

- `uv sync` creates or updates the environment from `pyproject.toml` and `uv.lock`.
- `uv run python example_h2.py` runs the small H2 inference and local-energy example.
- `uv run python train_vmc.py --epochs 5 --walkers 64` performs a short local training smoke test.
- `uv run python -m unittest discover -s . -p "test_*.py" -v` runs the complete test suite.
- `uv run python -m unittest test_vmc.py` runs one focused test module.
- `uv build` produces source and wheel distributions through Hatchling.

GPU-heavy benchmarks are optional; use `uv run python benchmark_100_atoms.py --double --quick` for the reduced path.

## Coding Style & Naming Conventions

Use Python 3.9+ and four-space indentation. Follow the existing PEP 8-like layout: `snake_case` for functions, variables, and modules; `PascalCase` for classes; and `UPPER_CASE` for constants. Add type annotations to public APIs and document tensor shapes, dtypes, and physical units where relevant. Preserve the established `(B, N, 3)` electron-coordinate convention. No formatter or linter is configured, so keep imports grouped, lines readable, and changes narrowly scoped.

## Testing Guidelines

Tests use the standard-library `unittest` framework. Name files `test_<area>.py`, classes `Test<Feature>`, and methods `test_<behavior>`. Seed PyTorch for stochastic cases and test physical invariants—shapes, finiteness, antisymmetry, and numerical tolerances—not only successful execution. Add regression coverage with every behavioral fix. The repository does not define a coverage threshold; the full suite must pass before review.

## Commit & Pull Request Guidelines

History follows Conventional Commit-style subjects such as `feat:`, `fix(visualize):`, `docs:`, and `style:`. Keep the subject imperative and describe one logical change. Pull requests should explain the motivation and numerical/physical impact, list commands run, and link related issues. Include updated plots or screenshots when visualization output changes, plus benchmark numbers when making performance claims. Do not commit virtual environments, checkpoints, logs, or generated build artifacts.
