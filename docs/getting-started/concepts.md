# Concepts

This page is a map of TSFast. After reading it you will know what components
the library provides and where to find them. For usage details, see the
[tutorials](../examples/notebooks/00_your_first_model.ipynb).

## What is System Identification?

System identification is the problem of learning a mathematical model of a
dynamical system from measured input-output data. Given an input signal
**u(t)** (e.g., voltage applied to a motor) and the measured output **y(t)**
(e.g., the motor's angular velocity), the goal is to train a model that
predicts **y** from **u**.

```
u(t)  ──►  [ System ]  ──►  y(t)
```

TSFast uses neural networks -- RNNs, TCNs, and hybrids -- as the learned
model. The training data consists of time series recordings of the real
system.

## Simulation vs. Prediction

This is the core conceptual distinction in TSFast. Every modeling choice flows
from this decision.

### Simulation

The model sees **only the input signal** u(t) and must predict y(t) with no
access to past measured outputs. This is harder but more general -- at
deployment time, you only need the input signal.

```
Model input:  [ u(t) ]
Model output: y(t)
```

### Prediction

The model receives **both** the input u(t) **and** past measured outputs
y(t-1). This is easier and typically more accurate, because the model can
correct itself using recent measurements. The tradeoff: you need a sensor
measuring y at deployment time.

```
Model input:  [ u(t),  y(t-1) ]
Model output: y(t)
```

## Library Overview

``` mermaid
graph LR
    A["Data Pipeline<br/><code>tsfast.tsdata</code>"]
    B["Training<br/><code>tsfast.training</code>"]
    C["Inference<br/><code>tsfast.inference</code>"]
    D["Models<br/><code>tsfast.models</code>"]
    E["Losses &amp; Transforms<br/><code>tsfast.training</code>"]

    A --> B
    D --> B
    E --> B
    B --> C
```

TSFast is organized around a pipeline with three stages:

- **Data Pipeline** loads time series from files, splits them into sliding
  windows, and builds DataLoaders.
- **Models** define the neural network architecture (RNN, TCN, or hybrid).
- **Training** wires model, data, losses, transforms, and optimizer into a
  training loop via the `Learner`.
- **Inference** wraps a trained model for numpy-based or ONNX deployment.

Three specialized modules extend the core: **prediction** (autoregressive and
FranSys models), **pinn** (physics-informed neural networks), and **tune**
(hyperparameter optimization).

## Data Pipeline

``` mermaid
graph LR
    F["HDF5 / CSV<br/>files"] --> R["Readers"]
    R --> W["WindowedDataset"]
    S["Splitting"] --> DL["DataLoaders"]
    N["Normalization"] --> DL
    W --> DL
```

The `create_dls` factory orchestrates the full pipeline: it discovers files,
applies a split strategy, creates windowed datasets, computes normalization
statistics, and returns `DataLoaders` ready for training.

### Readers

Readers extract signals from files. They are composable -- wrap any reader in
`Cached` or `Resampled` to add caching or on-the-fly resampling.

| Reader | Description |
|--------|-------------|
| `HDF5Signals` | Read named 1-D datasets from HDF5 files |
| `CSVSignals` | Read columns from CSV files |
| `HDF5Attrs` | Read scalar attributes from HDF5 metadata |
| `FilenameScalar` | Extract scalar values from filenames via regex |
| `Cached` | Wrapper: cache any reader's output in memory |
| `Resampled` | Wrapper: resample signals on-the-fly |

### Windowing

`WindowedDataset` creates many overlapping training samples from long time
series by sliding a window across each file. Key parameters: `win_sz` (window
length in samples) and `stp_sz` (step between windows).

### Splitting

| Function | Strategy |
|----------|----------|
| `split_by_percentage` | Random percentage-based train/valid/test split |
| `split_by_parent` | Split based on directory structure (`train/`, `valid/`, `test/`) |
| `discover_split_files` | Auto-detect strategy from file layout |

### Normalization

| Scaler | Description |
|--------|-------------|
| `StandardScaler` | z-score: (x - mean) / std |
| `MinMaxScaler` | Scale to [0, 1] range |
| `MaxAbsScaler` | Scale by max absolute value |

By default, **input signals are normalized** and **output signals stay in
physical units**. For multi-output systems with very different scales, enable
output normalization via `output_norm=StandardScaler` -- the `ScaledModel`
wrapper automatically denormalizes predictions back to physical units.

### Benchmark Datasets

`tsfast.tsdata.benchmark` provides one-line loaders for standard system
identification benchmarks via
[identibench](https://github.com/daniel-om-weber/identibench): Silverbox,
Wiener-Hammerstein, Cascaded Tanks, EMPS, CED, Robot Arm, Ship, and
Quadcopters. Each is available in simulation and prediction variants.

**Links:**
[API: Pipeline](../api/tsdata/pipeline.md) ·
[Readers](../api/tsdata/readers.md) ·
[Normalization](../api/tsdata/norm.md) ·
[Splitting](../api/tsdata/split.md) ·
[Benchmarks](../api/tsdata/benchmark.md) --
[Tutorial: Data Pipeline](../examples/notebooks/01_data_pipeline.ipynb) ·
[Custom Data Pipelines](../examples/notebooks/09_custom_data_pipelines.ipynb)

## Models

``` mermaid
graph TD
    M["Model Families"] --> RNN["RNN"]
    M --> CNN["CNN / TCN"]
    M --> HYB["Hybrid"]

    RNN --> SR["SimpleRNN"]
    RNN --> DR["DenseNet_RNN"]
    RNN --> RR["SimpleResidualRNN"]
    RNN --> SPR["SeperateRNN"]

    CNN --> TC["TCN"]
    CNN --> CN["CNN"]
    CNN --> SPT["SeperateTCN"]

    HYB --> CR["CRNN"]
    HYB --> SPC["SeperateCRNN"]
```

TSFast provides three families of sequence models. All use the
`[batch, seq_len, features]` tensor convention.

### Architectures

| Model | Family | Description |
|-------|--------|-------------|
| `SimpleRNN` | RNN | Multi-layer GRU or LSTM with linear output head |
| `DenseNet_RNN` | RNN | DenseNet-style feature concatenation across RNN layers |
| `SimpleResidualRNN` | RNN | Stack of residual RNN blocks |
| `SeperateRNN` | RNN | Per-channel-group RNNs merged before a final RNN |
| `TCN` | CNN | Temporal convolutional network with exponential dilation and causal padding |
| `CNN` | CNN | Stacked 1D convolutions |
| `SeperateTCN` | CNN | Per-group TCN branches with linear merge head |
| `CRNN` | Hybrid | TCN front-end feeding into an RNN back-end |
| `SeperateCRNN` | Hybrid | Per-group TCN branches merged before an RNN |

### Building Blocks and Wrappers

- `SeqLinear` -- pointwise MLP via 1x1 convolutions, applied at each timestep
- `AR_Model` -- autoregressive wrapper with teacher forcing during training
  and step-by-step inference
- `SeqAggregation` -- reduce the sequence dimension (last timestep, mean,
  etc.) for seq-to-scalar tasks
- `ScaledModel` -- wraps any model with input normalization and optional
  output denormalization
- `GraphedStatefulModel` -- CUDA graph capture for low-overhead GPU training
  of stateful models

**Links:**
[API: RNN](../api/models/rnn.md) ·
[CNN / TCN](../api/models/cnn.md) ·
[Layers](../api/models/layers.md) ·
[Scaling](../api/models/scaling.md) ·
[CUDA Graphs](../api/models/cudagraph.md) --
[Tutorial: Model Architectures](../examples/notebooks/04_model_architectures.ipynb) ·
[CUDA Graphs](../examples/notebooks/18_cuda_graphs.ipynb)

## Training

The `Learner` is the core training loop. It wraps a model, DataLoaders, loss
function, optimizer, metrics, transforms, and auxiliary losses. `TbpttLearner`
extends it for truncated backpropagation through time, carrying hidden state
across sub-windows of long sequences.

### Learner Factories

Factory functions create pre-configured Learners with sensible defaults --
they handle model creation, normalization wrapping, and default
loss/metrics/transforms.

| Factory | Model | Mode |
|---------|-------|------|
| `RNNLearner` | `SimpleRNN` | Simulation |
| `TCNLearner` | `TCN` | Simulation |
| `CRNNLearner` | `CRNN` | Simulation |
| `AR_RNNLearner` | `AR_Model` + `SimpleRNN` | Prediction (autoregressive) |
| `AR_TCNLearner` | `AR_Model` + `TCN` | Prediction (autoregressive) |

### Key Methods

- `fit(n_epoch, lr, cbs)` -- train with a custom schedule
- `fit_flat_cos(n_epoch, lr)` -- train with flat-then-cosine-annealing
  schedule (recommended default)
- `validate(dl)` -- compute loss and metrics on validation or test set
- `show_batch()` -- plot random training windows
- `show_results()` -- overlay predictions on validation data
- `save_model()` / `load()` -- persist and restore model weights
- `save()` / `load()` -- persist and restore full learner state

**Links:**
[API: Learner](../api/training/learner.md) --
[Tutorial: Your First Model](../examples/notebooks/00_your_first_model.ipynb) ·
[Saving and Loading](../examples/notebooks/08_saving_and_loading.ipynb) ·
[Stateful TBPTT](../examples/notebooks/17_stateful_tbptt.ipynb)

## Losses and Metrics

TSFast losses and metrics operate on `[batch, seq, features]` tensors.

### Core Functions

| Function | Description |
|----------|-------------|
| `mse` | Mean squared error |
| `fun_rmse` | Root mean squared error (default metric) |
| `nrmse` | RMSE normalized by target range |
| `nrmse_std` | RMSE normalized by target standard deviation |
| `weighted_mae` | Weighted mean absolute error |
| `mean_vaf` | Variance accounted for |
| `cos_sim_loss` | Cosine similarity loss |

### Wrappers

Wrappers compose with base loss functions to handle common scenarios.

| Wrapper | Description |
|---------|-------------|
| `nan_mean(fn)` | Make any loss NaN-safe via masked mean |
| `ignore_nan(fn)` | Skip NaN positions entirely |
| `cut_loss(fn, n_skip)` | Skip first *n* timesteps before computing loss |
| `norm_loss(fn, norm)` | Compute loss in normalized space |
| `rand_seq_len_loss(fn)` | Randomly truncate sequence length per batch |
| `float64_func(fn)` | Promote to float64 for numerical stability |

### Auxiliary Losses

- `ActivationRegularizer` -- L2 penalty on hidden activations (smoothness)
- `TemporalActivationRegularizer` -- L2 penalty on consecutive-timestep
  activation differences (temporal smoothness)
- `FranSysRegularizer` -- synchronizes diagnosis/prognosis hidden states

**Links:**
[API: Losses](../api/training/losses.md) ·
[Auxiliary Losses](../api/training/aux_losses.md) --
[Tutorial: Losses and Metrics](../examples/notebooks/05_losses_and_metrics.ipynb)

## Transforms and Augmentations

All transforms follow the `(xb, yb) -> (xb, yb)` protocol. **Transforms**
run on both train and validation data; **augmentations** run on train only.

| Name | Type | Description |
|------|------|-------------|
| `prediction_concat` | Transform | Concatenate past y onto x (enables prediction mode) |
| `truncate_sequence` | Transform | Progressively shorten sequences during training (curriculum) |
| `noise` | Augmentation | Add Gaussian noise with configurable std per signal |
| `noise_varying` | Augmentation | Add noise with randomly sampled std |
| `noise_grouped` | Augmentation | Add noise with per-group random std |
| `bias` | Augmentation | Add constant random offset per sample |
| `vary_seq_len` | Augmentation | Randomly vary sequence length per batch |

### Learning Rate Schedulers

- `sched_lin_p` -- linear schedule reaching target at position *p*
- `sched_ramp` -- linear ramp between two plateau regions

**Links:**
[API: Transforms](../api/training/transforms.md) ·
[Schedulers](../api/training/schedulers.md) --
[Tutorial: Augmentation and Regularization](../examples/notebooks/07_callbacks_and_augmentation.ipynb)

## Inference

After training, `InferenceWrapper` provides numpy-in/numpy-out inference with
the same preprocessing used during training. Models can also be exported to
ONNX for deployment without PyTorch.

- `InferenceWrapper` -- wrap a trained Learner for numpy-based inference
- `load_model()` -- load a saved model from disk
- `export_onnx()` -- export to ONNX format (normalization baked in)
- `OnnxInferenceWrapper` -- numpy-in/numpy-out ONNX inference

**Links:**
[API: Inference Wrapper](../api/inference/wrapper.md) ·
[ONNX](../api/inference/onnx.md) --
[Tutorial: ONNX Export](../examples/notebooks/16_onnx_export.ipynb) ·
[Saving and Loading](../examples/notebooks/08_saving_and_loading.ipynb)

## Specialized Modules

### FranSys (State Estimation)

FranSys models estimate the full internal state of a dynamical system from
partial observations. A *diagnosis* model (`Diag_RNN`, `Diag_TCN`) processes
an initialization window to estimate the hidden state, then a *prognosis*
model (`ARProg`, `ARProg_Init`) predicts forward autoregressively.
`FranSysLearner` provides a training loop with coordinated loss computation.

**Links:**
[API: FranSys](../api/prediction/fransys.md) ·
[Autoregressive](../api/prediction/ar.md) --
[Tutorial: FranSys](../examples/notebooks/10_fransys.ipynb) ·
[Autoregressive Models](../examples/notebooks/11_autoregressive.ipynb)

### Physics-Informed Neural Networks (PINN)

`PIRNN` embeds governing equations into training via physics losses computed
at collocation points. Auxiliary losses include `PhysicsLoss`,
`CollocationLoss`, `ConsistencyLoss`, and `TransitionSmoothnessLoss`. Signal
generation utilities produce diverse excitation signals for collocation data.

**Links:**
[API: PIRNN](../api/pinn/pirnn.md) ·
[Physics Losses](../api/pinn/aux_losses.md) ·
[Differentiation](../api/pinn/differentiation.md) ·
[Signal Generation](../api/pinn/signals.md) --
[Tutorial: Physics-Informed NNs](../examples/notebooks/12_pinn.ipynb)

### Hyperparameter Optimization

`HPOptimizer` integrates with Ray Tune for distributed hyperparameter search
with custom search spaces.

**Links:**
[API: HPOptimizer](../api/tune.md) --
[Tutorial: Hyperparameter Optimization](../examples/notebooks/15_hyperparameter_optimization.ipynb)

## Utility Modules

- **Quaternions** (`tsfast.quaternions`) -- quaternion algebra, orientation
  losses, and rotation augmentations for orientation estimation tasks.
  [API](../api/quaternions.md)
- **Spectrogram** (`tsfast.spectogram`) -- frequency-domain transforms via
  STFT. [API](../api/spectogram.md)

## Next Steps

- [Your First Model](../examples/notebooks/00_your_first_model.ipynb) --
  train an LSTM in under 10 lines
- [Data Pipeline](../examples/notebooks/01_data_pipeline.ipynb) -- understand
  what happens under the hood
- [Model Architectures](../examples/notebooks/04_model_architectures.ipynb) --
  compare RNNs, TCNs, and CRNNs
- [IdentiBench](../examples/notebooks/13_identibench.ipynb) -- benchmark on
  standard datasets
