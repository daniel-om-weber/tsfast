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

**A deep learning library for time series analysis and system identification, built on PyTorch.**

---

## Key Features

- **Specialized Data Handling** — `WindowedDataset`, sequence transforms, `TbpttLearner` for truncated backpropagation
- **Benchmark Datasets** — One-line access to Silverbox, Wiener-Hammerstein, Cascaded Tanks, and more via `identibench`
- **Time Series Models** — RNNs (DenseNet, Residual), TCNs, CRNNs with layer normalization
- **Integrated Training** — `RNNLearner`, `TCNLearner`, `CRNNLearner` with custom losses (`nrmse`, `cut_loss`) and transforms
- **System Identification** — Simulation, N-step prediction, FranSys, autoregressive models
- **Physics-Informed NNs** — Embed governing equations into training via `CollocationLoss`, `PhysicsLoss`, and `PIRNN`
- **Hyperparameter Optimization** — Ray Tune integration via `HPOptimizer`
- **Deployment** — `InferenceWrapper` for NumPy-in/NumPy-out inference, ONNX export, model save/load
- **Performance** — CUDA graph acceleration via `GraphedStatefulModel` for low-overhead GPU training

## Quick Start

```python
from tsfast.tsdata.benchmark import create_dls_silverbox
from tsfast.training import RNNLearner

# Load benchmark dataset
dls = create_dls_silverbox()

# Train an RNN and visualize results
lrn = RNNLearner(dls)
lrn.fit_flat_cos(n_epoch=1)
lrn.show_results(max_n=1)
```

Ready to learn more? Start with [Installation](getting-started/installation.md) or jump to [Your First Model](examples/notebooks/00_your_first_model.ipynb).

## Quick Import

TSFast provides a convenience barrel import for interactive use:

```python
from tsfast.basics import *
```

This imports all public symbols from `tsdata`, `training`, `models`, `prediction`, and `inference`. For production code, prefer explicit imports from specific modules (e.g., `from tsfast.training import RNNLearner`).

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
