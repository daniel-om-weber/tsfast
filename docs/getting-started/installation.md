# Installation

## Stable Release

Install the latest stable version from PyPI:

```bash
pip install tsfast
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv add tsfast
```

## Development Installation

```bash
git clone https://github.com/daniel-om-weber/tsfast
cd tsfast
uv sync --extra dev
```

This installs all development dependencies (pytest, ruff, jupytext, etc.) into a managed `.venv`.

## Optional: ONNX Support

For ONNX export and inference:

```bash
pip install tsfast[onnx]
```

Or with uv:

```bash
uv add tsfast --extra onnx
```

## Requirements

- Python >= 3.10
- PyTorch >= 2.1
- fastai >= 2.7.0
