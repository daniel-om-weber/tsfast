# CLAUDE.md

## Project Overview

TSFast is a deep learning library for time series analysis and system identification, built on PyTorch and fastai.

## Development Workflow

Edit `.py` files in `tsfast/` directly. Examples are in `examples/` as Jupyter notebooks. Integration tests are in `tests/` as pytest files.

## Commands

```bash
# Run all tests
pytest tests/ -v

# Run a specific test file
pytest tests/test_models.py -v

# Run example notebooks (verify they execute)
pytest --nbmake examples/00_minimal_example.ipynb

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
