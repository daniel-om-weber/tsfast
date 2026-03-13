

<img src="https://raw.githubusercontent.com/daniel-om-weber/tsfast/refs/heads/master/assets/logo.svg" width="200" align="left" alt="tsfast logo">

## TSFast

[![PyPI
version](https://badge.fury.io/py/tsfast.svg)](https://badge.fury.io/py/tsfast)
[![License: Apache
2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python
Versions](https://img.shields.io/pypi/pyversions/tsfast.png)](https://pypi.org/project/tsfast/)

## Description

> `tsfast` is an open-source deep learning library for time series
> analysis and system identification tasks. Built on PyTorch,
> it offers efficient deep learning models and utilities.

`tsfast` is an open-source deep learning package that focuses on system
identification and time series analysis tasks. Built on PyTorch, it
provides efficient implementations of various deep learning models and
utilities.

## Key Features

- **Specialized Data Handling** — HDF5-backed data pipeline with
  sequence transforms, noise injection, normalization, and TBPTT
  (Truncated Backpropagation Through Time) support.
- **Benchmark Datasets** — One-line access to Silverbox,
  Wiener-Hammerstein, Cascaded Tanks, and more via `identibench`.
- **Time Series Models** — RNNs (DenseNet, Residual), TCNs, CRNNs
  with layer normalization, and combined architectures.
- **Integrated Training** — `RNNLearner`, `TCNLearner`, `CRNNLearner`
  with custom losses (`nrmse`, `cut_loss`) and composable transforms.
- **System Identification** — Simulation, N-step prediction, FranSys,
  and autoregressive models with `InferenceWrapper` for deployment.
- **Physics-Informed NNs** — Embed governing equations into training
  via `CollocationLoss`, `PhysicsLoss`, and `PIRNN`.
- **Hyperparameter Optimization** — Ray Tune integration via
  `HPOptimizer`.
- **Deployment** — Model save/load, `InferenceWrapper` for
  NumPy-in/NumPy-out inference, and ONNX export.
- **Performance** — CUDA graph acceleration via `GraphedStatefulModel`
  for low-overhead GPU training.

## Installation

You can install the **latest stable** version using:

``` bash
pip install tsfast
```

For development installation:

``` bash
git clone https://github.com/daniel-om-weber/tsfast
cd tsfast
pip install -e '.[dev]'
# or using uv:
uv sync --extra dev
```

## Quick Start

Here is a quick example using a benchmark dataloader. It demonstrates
loading and visualizing data, training a RNN, and visualizing the
results.

``` python
from tsfast.basics import *

# Load benchmark dataset
dls = create_dls_silverbox()

# Train an RNN and visualize results
lrn = RNNLearner(dls)
lrn.fit_flat_cos(1)
lrn.show_results(max_n=1)
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
For major changes, please open an issue first to discuss what you would
like to change.

## Citation

If you use tsfast in your research, please cite:

``` text
@Misc{tsfast,
author = {Daniel O.M. Weber},
title = {tsfast - A deep learning library for time series analysis and system identification},
howpublished = {Github},
year = {2024},
url = {https://github.com/daniel-om-weber/tsfast}
}
```
