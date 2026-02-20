# CLAUDE.md

## Project Overview

TSFast is a deep learning library for time series analysis and system identification, built on PyTorch and fastai.

## Development Workflow

Edit `.py` files in `tsfast/` directly. Examples are in `examples/` as Jupyter notebooks. Integration tests are in `tests/` as pytest files.

## Commands

```bash
# Run all tests
pytest tests/ -v

# Run tests excluding slow training tests
pytest tests/ -v -m "not slow"

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

- Inline type hints: `param:type = default` with modern union syntax (`str|None`)
- Parameter descriptions as inline comments: `param:type, # description`
- Single-line docstrings under 80 chars; no parameter docs in docstrings
- Use `store_attr()` in constructors
- Use pattern matching (`match/case`) for type checking
