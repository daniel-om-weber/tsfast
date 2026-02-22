# CLAUDE.md

## Project Overview

TSFast is a deep learning library for time series analysis and system identification, built on PyTorch and fastai.

## Development Workflow

Edit `.py` files in `tsfast/` directly. Examples are in `examples/` as jupytext-paired notebooks. Integration tests are in `tests/` as pytest files.

## Example Notebooks

Examples use jupytext pairing with separate directories:

- `examples/scripts/` — `.py` percent-format files (source of truth for diffs/review)
- `examples/notebooks/` — `.ipynb` files with rendered outputs (for browsing)
- `examples/04_benchmark_rnn.py` — standalone script, not a paired notebook

A pre-commit hook syncs the pair automatically on commit. Edit either file.

```bash
# Manually sync all paired notebooks
cd examples && jupytext --sync notebooks/*.ipynb

# Convert a new notebook to paired format
cd examples && jupytext --set-formats notebooks//ipynb,scripts//py:percent notebooks/new_notebook.ipynb
```

## Commands

```bash
# Run all tests
pytest tests/ -v

# Run a specific test file
pytest tests/test_models.py -v

# Run example notebooks (verify they execute)
pytest --nbmake examples/notebooks/00_minimal_example.ipynb

# Run all example notebooks and save outputs in-place (takes a long time)
jupyter nbconvert --to notebook --execute --inplace examples/notebooks/*.ipynb

# Lint
ruff check tsfast/

# Format
ruff format tsfast/

# Install/sync dependencies (uses uv.lock)
uv sync --extra dev
```

## Environment

Use `uv sync --extra dev` to install dependencies — it creates/manages the `.venv` automatically and respects `uv.lock` for reproducible installs. Ruff is configured in `pyproject.toml` to ignore F403, F405 (wildcard imports) and E702 (multiple statements on one line).


## Code Style

- Inline type hints on all public API signatures: `param:type = default` with modern union syntax (`str|None`)
- Google-style docstrings — types belong in the signature, never duplicated in `Args:`
- Module docstrings: one-liner stating what the module provides
- Class docstrings: short description + `Args:` for `__init__` params (describe meaning, not types)
- Skip `Args:` for obvious signatures (`forward(self, x)`, `__call__`)
- `Returns:` only when non-obvious from type hint
- One-liner docstrings are fine for simple utility functions
- No docstrings needed for: private helpers, fastai callback lifecycle methods, trivial property accessors
- Use pattern matching (`match/case`) for type checking
