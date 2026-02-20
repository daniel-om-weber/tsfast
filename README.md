

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
> analysis and system identification tasks. Built on PyTorch & fastai,
> it offers efficient deep learning models and utilities.

`tsfast` is an open-source deep learning package that focuses on system
identification and time series analysis tasks. Built on the foundations
of PyTorch and fastai, it provides efficient implementations of various
deep learning models and utilities.

## Key Features

- **Specialized Data Handling for Time Series**:
  - Employs `SequenceBlock` (built on `fastai.TransformBlock`) for
    robust sequence data processing pipelines.
  - Includes a range of transforms tailored for sequences, such as
    `SeqSlice`, `SeqNoiseInjection`, and `Normalize` adapted for time
    series tensors.
  - Features advanced data loading with `TbpttDl` (for Truncated
    Backpropagation Through Time), and factories for `WeightedDL` and
    `BatchLimitDL`.
- **Predefined Datasets & Helpers**: Offers easy-to-use benchmark
  datasets (e.g., `create_dls_silverbox` from `identibench`) for rapid
  prototyping and experimentation.
- **Tailored Time Series Models**: Provides implementations of Recurrent
  Neural Networks (RNNs, including `DenseNet_RNN`, `ResidualBlock_RNN`),
  Convolutional Neural Networks (TCNs, `CausalConv1d`),
  and combined architectures (`CRNN`, `SeperateCRNN`)
  specifically designed for sequence modeling. Includes building blocks
  like `SeqLinear` and stateful batch normalization.
- **Integrated `fastai` Learner**: Features `RNNLearner`, `TCNLearner`,
  `CRNNLearner`, etc., extending `fastai`’s `Learner` for streamlined
  model training, equipped with custom time-series losses (e.g.,
  `fun_rmse`, `nrmse`) and callbacks (e.g., `TbpttResetCB`, `ARInitCB`,
  `SkipFirstNCallback`).
- **System Identification & Prediction**:
  - Supports simulation (prediction based on inputs) and N-step ahead
    forecasting.
  - Includes specialized models and callbacks for system identification
    tasks like FRANSYS (`FranSys`, `FranSysCallback`) and AR models
    (`AR_Model`, `ARProg`).
  - Provides an `InferenceWrapper` for easier model deployment and
    prediction.
- **Hyperparameter Optimization**: Integrates with Ray Tune via
  `HPOptimizer` for efficient hyperparameter searching.

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

# Load benchmark dataset and visualize
dls = create_dls_silverbox()
dls.show_batch(max_n=1)

# Train an RNN
lrn = RNNLearner(dls)
lrn.fit_flat_cos(1)

# Visualize results
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
