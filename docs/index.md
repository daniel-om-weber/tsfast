---
hide:
  - navigation
---

<div style="text-align: center;">
  <img src="assets/logo.svg" width="200" alt="TSFast logo">
</div>

# TSFast

[![PyPI version](https://badge.fury.io/py/tsfast.svg)](https://badge.fury.io/py/tsfast)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python Versions](https://img.shields.io/pypi/pyversions/tsfast.png)](https://pypi.org/project/tsfast/)

**A deep learning library for time series analysis and system identification, built on PyTorch and fastai.**

---

## Key Features

- **Specialized Data Handling** — `SequenceBlock`, sequence transforms, `TbpttDl` for truncated backpropagation
- **Benchmark Datasets** — One-line access to Silverbox, Wiener-Hammerstein, Cascaded Tanks, and more via `identibench`
- **Time Series Models** — RNNs (DenseNet, Residual), TCNs, CRNNs with stateful batch normalization
- **Integrated Training** — `RNNLearner`, `TCNLearner`, `CRNNLearner` with custom losses (`nrmse`, `SkipNLoss`) and callbacks
- **System Identification** — Simulation, N-step prediction, FranSys, autoregressive models
- **Hyperparameter Optimization** — Ray Tune integration via `HPOptimizer`
- **Deployment** — `InferenceWrapper` for NumPy-in/NumPy-out inference, ONNX export

## Quick Start

```python
from tsfast.basics import *

# Load benchmark dataset and visualize
dls = create_dls_silverbox()
dls.show_batch(max_n=1)

# Train an RNN
lrn = RNNLearner(dls)
lrn.fit_flat_cos(1)

# Visualize results
lrn.show_results(max_n=1)
```

Ready to learn more? Start with [Installation](getting-started/installation.md) or jump to [Your First Model](examples/notebooks/00_your_first_model.ipynb).

## Quick Import

TSFast provides a convenience barrel import for interactive use:

```python
from tsfast.basics import *
```

This imports all public symbols from `data`, `datasets`, `models`, `learner`, `prediction`, and `inference`. For production code, prefer explicit imports from specific modules (e.g., `from tsfast.models.rnn import RNNLearner`).

## Citation

If you use TSFast in your research, please cite:

```text
@Misc{tsfast,
author = {Daniel O.M. Weber},
title = {tsfast - A deep learning library for time series analysis and system identification},
howpublished = {Github},
year = {2024},
url = {https://github.com/daniel-om-weber/tsfast}
}
```
