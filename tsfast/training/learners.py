"""Factory functions that wire up models with training configuration."""

__all__ = [
    "RNNLearner",
    "AR_RNNLearner",
    "TCNLearner",
    "CRNNLearner",
    "AR_TCNLearner",
    "SSMLearner",
    "DynoNetLearner",
    "NarxMLPLearner",
    "LRULearner",
    "S5Learner",
    "MambaLearner",
]

import torch
from torch import nn

from .aux_losses import ActivationRegularizer, TemporalActivationRegularizer
from .learner import Learner, TbpttLearner
from .losses import fun_rmse
from .transforms import prediction_concat
from ..models.cudagraph import GraphedStatefulModel
from ..models.cnn import CRNN, TCN
from ..models.dynonet import DynoNet
from ..models.layers import AR_Model
from ..models.lru import DeepLRU
from ..models.mamba import DeepMamba
from ..models.s5 import DeepS5
from ..models.narx import NarxMLP
from ..models.rnn import SimpleRNN
from ..models.ssm import NeuralStateSpace
from ..models.scaling import ScaledModel, Scaler, StandardScaler
from ..tsdata import get_io_size


def RNNLearner(
    dls,
    loss_func=nn.L1Loss(),
    metrics: list | None = None,
    lr: float = 3e-3,
    n_skip: int = 0,
    num_layers: int = 1,
    hidden_size: int = 100,
    sub_seq_len: int | None = None,
    opt_func=torch.optim.Adam,
    input_norm: type | None = StandardScaler,
    output_norm: type | None = None,
    augmentations: list | None = None,
    transforms: list | None = None,
    aux_losses: list | None = None,
    grad_clip: float | None = None,
    plot_fn=None,
    cuda_graph: bool = False,
    device: torch.device | None = None,
    show_bar: bool = True,
    **kwargs,
):
    """Create a Learner with a SimpleRNN model and standard training setup.

    Args:
        dls: DataLoaders providing training and validation data.
        loss_func: loss function for training.
        metrics: list of metric functions evaluated during validation.
        n_skip: number of initial timesteps to skip in loss and metric computation.
        num_layers: number of stacked RNN layers.
        hidden_size: number of hidden units per RNN layer.
        sub_seq_len: sub-sequence length for TBPTT; enables stateful training when set.
        opt_func: optimizer constructor.
        input_norm: scaler class for input normalization, or None to disable.
        output_norm: scaler class for output denormalization, or None to disable.
        augmentations: list of augmentation transforms (train only).
        transforms: list of transforms (train + valid).
        aux_losses: list of auxiliary loss functions.
        grad_clip: max gradient norm for clipping, or None to disable.
        plot_fn: plotting function for show_batch/show_results.
        cuda_graph: if True and sub_seq_len is set, wrap the model in GraphedStatefulModel for faster training.
        device: target device (auto-detected if None).
        show_bar: whether to show tqdm progress bars.
        **kwargs: additional keyword arguments forwarded to ``SimpleRNN``.
    """
    if metrics is None:
        metrics = [fun_rmse]

    inp, out = get_io_size(dls)
    if sub_seq_len:
        kwargs.setdefault("return_state", True)
    model = SimpleRNN(inp, out, num_layers, hidden_size, **kwargs)
    model = ScaledModel.from_dls(model, dls, input_norm, output_norm)

    if sub_seq_len and cuda_graph:
        model = GraphedStatefulModel(model)
    cls = TbpttLearner if sub_seq_len else Learner
    extra = {"sub_seq_len": sub_seq_len} if sub_seq_len else {}
    return cls(
        model,
        dls,
        loss_func=loss_func,
        metrics=metrics,
        lr=lr,
        n_skip=n_skip,
        opt_func=opt_func,
        augmentations=augmentations,
        transforms=transforms,
        aux_losses=aux_losses,
        grad_clip=grad_clip,
        plot_fn=plot_fn,
        device=device,
        show_bar=show_bar,
        **extra,
    )


def AR_RNNLearner(
    dls,
    alpha: float = 0,
    beta: float = 0,
    loss_func=nn.L1Loss(),
    metrics: list | None = None,
    lr: float = 3e-3,
    n_skip: int = 0,
    opt_func=torch.optim.Adam,
    input_norm: type | None = StandardScaler,
    transforms: list | None = None,
    augmentations: list | None = None,
    aux_losses: list | None = None,
    grad_clip: float | None = None,
    plot_fn=None,
    device: torch.device | None = None,
    show_bar: bool = True,
    **kwargs,
):
    """Create a Learner with an autoregressive RNN model.

    Args:
        dls: DataLoaders providing training and validation data.
        alpha: activation regularization penalty weight.
        beta: temporal activation regularization penalty weight.
        loss_func: loss function for training.
        metrics: metric functions for validation, or None for default RMSE.
        lr: learning rate.
        n_skip: number of initial timesteps to skip in metric computation.
        opt_func: optimizer constructor.
        input_norm: scaler class for input normalization, or None to disable.
        transforms: list of transforms (train + valid); defaults to prediction_concat.
        augmentations: list of augmentation transforms (train only).
        aux_losses: list of auxiliary loss functions; defaults to activation regularizers.
        grad_clip: max gradient norm for clipping, or None to disable.
        plot_fn: plotting function for show_batch/show_results.
        device: target device (auto-detected if None).
        show_bar: whether to show tqdm progress bars.
        **kwargs: additional keyword arguments forwarded to ``SimpleRNN``.
    """
    if metrics is None:
        metrics = [fun_rmse]

    if transforms is None:
        transforms = [prediction_concat(t_offset=0)]

    inp, out = get_io_size(dls)
    ar_model = AR_Model(SimpleRNN(inp + out, out, **kwargs), ar=False)

    if aux_losses is None:
        rnn_module = ar_model.model.rnn
        aux_losses = [
            ActivationRegularizer(modules=[rnn_module], alpha=alpha),
            TemporalActivationRegularizer(modules=[rnn_module], beta=beta),
        ]

    model = ScaledModel.from_dls(ar_model, dls, input_norm, autoregressive=True)

    return Learner(
        model,
        dls,
        loss_func=loss_func,
        metrics=metrics,
        lr=lr,
        n_skip=n_skip,
        opt_func=opt_func,
        transforms=transforms,
        augmentations=augmentations,
        aux_losses=aux_losses,
        grad_clip=grad_clip,
        plot_fn=plot_fn,
        device=device,
        show_bar=show_bar,
    )


def SSMLearner(
    dls,
    n_state: int = 8,
    hidden_size: int | list[int] = 64,
    num_layers: int = 2,
    act: str = "tanh",
    backend: str = "auto",
    loss_func=nn.MSELoss(),
    metrics: list | None = None,
    lr: float = 3e-3,
    n_skip: int = 0,
    sub_seq_len: int | None = None,
    opt_func=torch.optim.Adam,
    input_norm: type | None = StandardScaler,
    output_norm: type | None = StandardScaler,
    transforms: list | None = None,
    augmentations: list | None = None,
    aux_losses: list | None = None,
    grad_clip: float | None = None,
    plot_fn=None,
    cuda_graph: bool = False,
    device: torch.device | None = None,
    show_bar: bool = True,
    **kwargs,
):
    """Create a Learner with a NeuralStateSpace model ``x_{k+1} = f(x_k, u_k)``, ``y_k = C x_k + d``.

    The latent state dimension ``n_state`` is independent of the dataset's output size and
    must be at least the order of the system being identified — a state narrower than the
    dynamics (e.g. a scalar state for a resonant second-order system) cannot carry the
    velocity/phase information the rollout needs and will not train past a constant predictor.

    The model rolls out from a zero initial state (in normalized coordinates), so use
    ``n_skip`` to exclude the initial transient from the loss when the true initial state
    is unknown. With ``sub_seq_len`` the physical state is carried across chunks (TBPTT),
    which limits the cold start to the first chunk of each batch. The fused ``c``/``triton``
    backends make the sequential rollout 1–2 orders of magnitude faster to train than the
    naive loop; see ``NeuralStateSpace``.

    Args:
        dls: DataLoaders providing training and validation data.
        n_state: latent state dimension of the state space model.
        hidden_size: hidden width, or an explicit list of hidden widths.
        num_layers: number of hidden layers (ignored when ``hidden_size`` is a list).
        act: transition MLP activation (``tanh``/``sigmoid``/``relu``).
        backend: execution backend of the rollout, see ``NeuralStateSpace``.
        loss_func: loss function for training.
        metrics: metric functions for validation, or None for default RMSE.
        n_skip: number of initial timesteps to skip in loss and metric computation.
        sub_seq_len: sub-sequence length for TBPTT; enables stateful training when set.
        opt_func: optimizer constructor.
        input_norm: scaler class for input normalization, or None to disable.
        output_norm: scaler class for output denormalization, or None to disable.
        transforms: list of transforms (train + valid).
        augmentations: list of augmentation transforms (train only).
        aux_losses: list of auxiliary loss functions.
        grad_clip: max gradient norm for clipping, or None to disable.
        plot_fn: plotting function for show_batch/show_results.
        cuda_graph: if True and sub_seq_len is set, wrap the model in GraphedStatefulModel
            (captures the model forward+backward; requires fixed batch/chunk shapes).
        device: target device (auto-detected if None).
        show_bar: whether to show tqdm progress bars.
        **kwargs: additional keyword arguments forwarded to ``NeuralStateSpace``.
    """
    if metrics is None:
        metrics = [fun_rmse]

    inp, out = get_io_size(dls)
    if sub_seq_len:
        kwargs.setdefault("return_state", True)
    model = NeuralStateSpace(inp, out, n_state, hidden_size, num_layers, act=act, backend=backend, **kwargs)
    model = ScaledModel.from_dls(model, dls, input_norm, output_norm)

    if sub_seq_len and cuda_graph:
        model = GraphedStatefulModel(model)
    cls = TbpttLearner if sub_seq_len else Learner
    extra = {"sub_seq_len": sub_seq_len} if sub_seq_len else {}
    return cls(
        model,
        dls,
        loss_func=loss_func,
        metrics=metrics,
        lr=lr,
        n_skip=n_skip,
        opt_func=opt_func,
        transforms=transforms,
        augmentations=augmentations,
        aux_losses=aux_losses,
        grad_clip=grad_clip,
        plot_fn=plot_fn,
        device=device,
        show_bar=show_bar,
        **extra,
    )


def DynoNetLearner(
    dls,
    n_channels: int = 8,
    nb: int = 8,
    na: int = 2,
    hidden_size: int = 32,
    hidden_layers: int = 1,
    bypass: bool = True,
    backend: str = "scan",
    loss_func=nn.MSELoss(),
    metrics: list | None = None,
    lr: float = 3e-3,
    n_skip: int = 0,
    sub_seq_len: int | None = None,
    opt_func=torch.optim.Adam,
    input_norm: type | None = StandardScaler,
    output_norm: type | None = StandardScaler,
    transforms: list | None = None,
    augmentations: list | None = None,
    aux_losses: list | None = None,
    grad_clip: float | None = None,
    plot_fn=None,
    device: torch.device | None = None,
    show_bar: bool = True,
    **kwargs,
):
    """Create a Learner with a DynoNet model: linear transfer functions G(q) around a static nonlinearity.

    The blocks have no finite receptive field: the FIR part settles after ``nb - 1`` samples,
    but the IIR transient decays with the slowest learned pole, which is unknown a priori.
    Set ``n_skip`` to the task's initial-condition window (or a few multiples of the dominant
    time constant) to keep the cold-start transient out of the loss. Coefficients are
    unconstrained, so poles can drift outside the unit circle during training; the Learner's
    NaN guard skips the affected steps and ``grad_clip`` damps the gradient spikes near the
    stability boundary.

    Args:
        dls: DataLoaders providing training and validation data.
        n_channels: signal width between the G-blocks.
        nb: numerator taps per filter in every G-block.
        na: denominator order per filter in every G-block.
        hidden_size: hidden width of the static nonlinearity MLP.
        hidden_layers: number of hidden layers of the static nonlinearity MLP.
        bypass: add a parallel linear path from input to output.
        backend: execution backend of the G-blocks, see ``LinearDynamicalOperator``.
        loss_func: loss function for training.
        metrics: metric functions for validation, or None for default RMSE.
        n_skip: number of initial timesteps to skip in loss and metric computation.
        sub_seq_len: sub-sequence length for TBPTT; enables stateful training when set.
        opt_func: optimizer constructor.
        input_norm: scaler class for input normalization, or None to disable.
        output_norm: scaler class for output denormalization, or None to disable.
        transforms: list of transforms (train + valid).
        augmentations: list of augmentation transforms (train only).
        aux_losses: list of auxiliary loss functions.
        grad_clip: max gradient norm for clipping, or None to disable.
        plot_fn: plotting function for show_batch/show_results.
        device: target device (auto-detected if None).
        show_bar: whether to show tqdm progress bars.
        **kwargs: additional keyword arguments forwarded to ``DynoNet``.
    """
    if metrics is None:
        metrics = [fun_rmse]

    inp, out = get_io_size(dls)
    if sub_seq_len:
        kwargs.setdefault("return_state", True)
    model = DynoNet(
        inp,
        out,
        n_channels=n_channels,
        nb=nb,
        na=na,
        hidden_size=hidden_size,
        hidden_layers=hidden_layers,
        bypass=bypass,
        backend=backend,
        **kwargs,
    )
    model = ScaledModel.from_dls(model, dls, input_norm, output_norm)

    cls = TbpttLearner if sub_seq_len else Learner
    extra = {"sub_seq_len": sub_seq_len} if sub_seq_len else {}
    return cls(
        model,
        dls,
        loss_func=loss_func,
        metrics=metrics,
        lr=lr,
        n_skip=n_skip,
        opt_func=opt_func,
        transforms=transforms,
        augmentations=augmentations,
        aux_losses=aux_losses,
        grad_clip=grad_clip,
        plot_fn=plot_fn,
        device=device,
        show_bar=show_bar,
        **extra,
    )


def LRULearner(
    dls,
    d_model: int = 32,
    d_state: int = 64,
    n_layers: int = 3,
    dropout: float = 0.0,
    r_min: float = 0.0,
    r_max: float = 1.0,
    max_phase: float = 6.283,
    backend: str = "scan",
    loss_func=nn.MSELoss(),
    metrics: list | None = None,
    lr: float = 3e-3,
    n_skip: int = 0,
    sub_seq_len: int | None = None,
    opt_func=torch.optim.Adam,
    input_norm: type | None = StandardScaler,
    output_norm: type | None = StandardScaler,
    transforms: list | None = None,
    augmentations: list | None = None,
    aux_losses: list | None = None,
    grad_clip: float | None = None,
    plot_fn=None,
    device: torch.device | None = None,
    show_bar: bool = True,
    **kwargs,
):
    """Create a Learner with a DeepLRU model: stacked diagonal linear state-space layers.

    Every block is a linear time-invariant system with a static nonlinearity between
    blocks, so the stack is a deep Wiener-Hammerstein-like cascade with stable poles by
    construction. The recurrence transient decays with the slowest learned eigenvalue;
    set ``n_skip`` to the task's initial-condition window to keep the cold start out of
    the loss. ``r_min``/``r_max`` bound the eigenvalue magnitudes at initialization only —
    slow dominant dynamics benefit from a ring close to 1 (e.g. ``[0.9, 0.999]``).

    Args:
        dls: DataLoaders providing training and validation data.
        d_model: signal width between the blocks.
        d_state: complex state dimension per block.
        n_layers: number of LRU blocks.
        dropout: dropout probability inside the blocks.
        r_min: eigenvalue-ring lower bound at initialization.
        r_max: eigenvalue-ring upper bound at initialization.
        max_phase: eigenvalue-phase upper bound at initialization (radians).
        backend: execution backend of the recurrences, see ``LRU``.
        loss_func: loss function for training.
        metrics: metric functions for validation, or None for default RMSE.
        n_skip: number of initial timesteps to skip in loss and metric computation.
        sub_seq_len: sub-sequence length for TBPTT; enables stateful training when set.
        opt_func: optimizer constructor.
        input_norm: scaler class for input normalization, or None to disable.
        output_norm: scaler class for output denormalization, or None to disable.
        transforms: list of transforms (train + valid).
        augmentations: list of augmentation transforms (train only).
        aux_losses: list of auxiliary loss functions.
        grad_clip: max gradient norm for clipping, or None to disable.
        plot_fn: plotting function for show_batch/show_results.
        device: target device (auto-detected if None).
        show_bar: whether to show tqdm progress bars.
        **kwargs: additional keyword arguments forwarded to ``DeepLRU``.
    """
    if metrics is None:
        metrics = [fun_rmse]

    inp, out = get_io_size(dls)
    if sub_seq_len:
        kwargs.setdefault("return_state", True)
    model = DeepLRU(
        inp,
        out,
        d_model=d_model,
        d_state=d_state,
        n_layers=n_layers,
        dropout=dropout,
        r_min=r_min,
        r_max=r_max,
        max_phase=max_phase,
        backend=backend,
        **kwargs,
    )
    model = ScaledModel.from_dls(model, dls, input_norm, output_norm)

    cls = TbpttLearner if sub_seq_len else Learner
    extra = {"sub_seq_len": sub_seq_len} if sub_seq_len else {}
    return cls(
        model,
        dls,
        loss_func=loss_func,
        metrics=metrics,
        lr=lr,
        n_skip=n_skip,
        opt_func=opt_func,
        transforms=transforms,
        augmentations=augmentations,
        aux_losses=aux_losses,
        grad_clip=grad_clip,
        plot_fn=plot_fn,
        device=device,
        show_bar=show_bar,
        **extra,
    )


def S5Learner(
    dls,
    d_model: int = 32,
    d_state: int = 64,
    n_layers: int = 3,
    blocks: int = 1,
    dropout: float = 0.0,
    activation: str = "half_glu1",
    dt_min: float = 0.001,
    dt_max: float = 0.1,
    backend: str = "scan",
    loss_func=nn.MSELoss(),
    metrics: list | None = None,
    lr: float = 3e-3,
    n_skip: int = 0,
    sub_seq_len: int | None = None,
    opt_func=torch.optim.Adam,
    input_norm: type | None = StandardScaler,
    output_norm: type | None = StandardScaler,
    transforms: list | None = None,
    augmentations: list | None = None,
    aux_losses: list | None = None,
    grad_clip: float | None = None,
    plot_fn=None,
    device: torch.device | None = None,
    show_bar: bool = True,
    **kwargs,
):
    """Create a Learner with a DeepS5 model: stacked HiPPO-initialized diagonal SSM layers.

    Like the LRU, every block is a linear time-invariant system with a static nonlinearity
    between blocks; the difference is the continuous-time parameterization with per-state
    learned timesteps, ZOH-discretized every forward pass. ``dt_min``/``dt_max`` set the
    timescale range at initialization relative to the sampling time — widen towards small
    values for fast dynamics, large for slow. Set ``n_skip`` to the task's initial-condition
    window to keep the cold-start transient out of the loss.

    Args:
        dls: DataLoaders providing training and validation data.
        d_model: signal width between the blocks.
        d_state: full state dimension per block.
        n_layers: number of S5 blocks.
        blocks: HiPPO blocks per layer at initialization.
        dropout: dropout probability inside the blocks.
        activation: block activation variant, see ``S5Block``.
        dt_min: timestep initialization lower bound.
        dt_max: timestep initialization upper bound.
        backend: execution backend of the recurrences, see ``S5``.
        loss_func: loss function for training.
        metrics: metric functions for validation, or None for default RMSE.
        n_skip: number of initial timesteps to skip in loss and metric computation.
        sub_seq_len: sub-sequence length for TBPTT; enables stateful training when set.
        opt_func: optimizer constructor.
        input_norm: scaler class for input normalization, or None to disable.
        output_norm: scaler class for output denormalization, or None to disable.
        transforms: list of transforms (train + valid).
        augmentations: list of augmentation transforms (train only).
        aux_losses: list of auxiliary loss functions.
        grad_clip: max gradient norm for clipping, or None to disable.
        plot_fn: plotting function for show_batch/show_results.
        device: target device (auto-detected if None).
        show_bar: whether to show tqdm progress bars.
        **kwargs: additional keyword arguments forwarded to ``DeepS5``.
    """
    if metrics is None:
        metrics = [fun_rmse]

    inp, out = get_io_size(dls)
    if sub_seq_len:
        kwargs.setdefault("return_state", True)
    model = DeepS5(
        inp,
        out,
        d_model=d_model,
        d_state=d_state,
        n_layers=n_layers,
        blocks=blocks,
        dropout=dropout,
        activation=activation,
        dt_min=dt_min,
        dt_max=dt_max,
        backend=backend,
        **kwargs,
    )
    model = ScaledModel.from_dls(model, dls, input_norm, output_norm)

    cls = TbpttLearner if sub_seq_len else Learner
    extra = {"sub_seq_len": sub_seq_len} if sub_seq_len else {}
    return cls(
        model,
        dls,
        loss_func=loss_func,
        metrics=metrics,
        lr=lr,
        n_skip=n_skip,
        opt_func=opt_func,
        transforms=transforms,
        augmentations=augmentations,
        aux_losses=aux_losses,
        grad_clip=grad_clip,
        plot_fn=plot_fn,
        device=device,
        show_bar=show_bar,
        **extra,
    )


def MambaLearner(
    dls,
    d_model: int = 32,
    d_state: int = 16,
    n_layers: int = 3,
    d_conv: int = 4,
    expand: int = 2,
    dt_min: float = 0.001,
    dt_max: float = 0.1,
    backend: str = "scan",
    loss_func=nn.MSELoss(),
    metrics: list | None = None,
    lr: float = 3e-3,
    n_skip: int = 0,
    sub_seq_len: int | None = None,
    opt_func=torch.optim.Adam,
    input_norm: type | None = StandardScaler,
    output_norm: type | None = StandardScaler,
    transforms: list | None = None,
    augmentations: list | None = None,
    aux_losses: list | None = None,
    grad_clip: float | None = None,
    plot_fn=None,
    device: torch.device | None = None,
    show_bar: bool = True,
    **kwargs,
):
    """Create a Learner with a DeepMamba model: stacked selective state-space blocks.

    Unlike the LRU/S5 layers, the state-space parameters are input-dependent, so each block
    is a structured nonlinear state-space model that can gate and switch dynamics with the
    signal. ``d_state``, ``d_conv``, and ``expand`` are architecture constants in the
    reference and rarely need tuning. Set ``n_skip`` to the task's initial-condition window
    to keep the cold-start transient out of the loss.

    Args:
        dls: DataLoaders providing training and validation data.
        d_model: signal width between the blocks.
        d_state: SSM state dimension per channel.
        n_layers: number of Mamba blocks.
        d_conv: depthwise convolution kernel width.
        expand: inner width multiplier.
        dt_min: timestep initialization lower bound.
        dt_max: timestep initialization upper bound.
        backend: execution backend of the scans, see ``MambaLayer``.
        loss_func: loss function for training.
        metrics: metric functions for validation, or None for default RMSE.
        n_skip: number of initial timesteps to skip in loss and metric computation.
        sub_seq_len: sub-sequence length for TBPTT; enables stateful training when set.
        opt_func: optimizer constructor.
        input_norm: scaler class for input normalization, or None to disable.
        output_norm: scaler class for output denormalization, or None to disable.
        transforms: list of transforms (train + valid).
        augmentations: list of augmentation transforms (train only).
        aux_losses: list of auxiliary loss functions.
        grad_clip: max gradient norm for clipping, or None to disable.
        plot_fn: plotting function for show_batch/show_results.
        device: target device (auto-detected if None).
        show_bar: whether to show tqdm progress bars.
        **kwargs: additional keyword arguments forwarded to ``DeepMamba``.
    """
    if metrics is None:
        metrics = [fun_rmse]

    inp, out = get_io_size(dls)
    if sub_seq_len:
        kwargs.setdefault("return_state", True)
    model = DeepMamba(
        inp,
        out,
        d_model=d_model,
        d_state=d_state,
        n_layers=n_layers,
        d_conv=d_conv,
        expand=expand,
        dt_min=dt_min,
        dt_max=dt_max,
        backend=backend,
        **kwargs,
    )
    model = ScaledModel.from_dls(model, dls, input_norm, output_norm)

    cls = TbpttLearner if sub_seq_len else Learner
    extra = {"sub_seq_len": sub_seq_len} if sub_seq_len else {}
    return cls(
        model,
        dls,
        loss_func=loss_func,
        metrics=metrics,
        lr=lr,
        n_skip=n_skip,
        opt_func=opt_func,
        transforms=transforms,
        augmentations=augmentations,
        aux_losses=aux_losses,
        grad_clip=grad_clip,
        plot_fn=plot_fn,
        device=device,
        show_bar=show_bar,
        **extra,
    )


def NarxMLPLearner(
    dls,
    na: int = 8,
    nb: int = 8,
    hidden_size: int = 64,
    num_layers: int = 2,
    train_mode: str = "free_run",
    washout: int | None = None,
    loss_func=nn.MSELoss(),
    metrics: list | None = None,
    lr: float = 3e-3,
    n_skip: int | None = None,
    opt_func=torch.optim.Adam,
    input_norm: type | None = StandardScaler,
    transforms: list | None = None,
    augmentations: list | None = None,
    aux_losses: list | None = None,
    grad_clip: float | None = None,
    plot_fn=None,
    device: torch.device | None = None,
    show_bar: bool = True,
    **kwargs,
):
    """Create a Learner with an MLP NARX model over explicit lag windows.

    ``train_mode`` selects how the output lags are filled during training:
    ``"free_run"`` feeds back the model's own predictions (simulation training,
    gradients flow through the feedback), ``"one_step"`` reads the true outputs
    (teacher forcing, fully parallel). Validation losses and inference always
    free-run, so the two modes stay comparable on the objective that matters
    and ``valid_loss`` is a simulation error in both cases.

    Args:
        dls: DataLoaders providing training and validation data.
        na: output lags per channel (autoregressive order).
        nb: input lags per channel, including the current sample.
        hidden_size: width of the hidden layers.
        num_layers: number of hidden layers.
        train_mode: ``"free_run"`` or ``"one_step"``.
        washout: initial samples whose true outputs seed the lag buffer in free
            run; defaults to ``n_skip`` so the teacher-forced prefix never enters
            the loss.
        loss_func: loss function for training.
        metrics: metric functions for validation, or None for default RMSE.
        lr: learning rate.
        n_skip: number of initial timesteps to skip in loss and metric
            computation; defaults to the lag-window length ``max(na, nb - 1)``.
        opt_func: optimizer constructor.
        input_norm: scaler class for the concatenated ``[u, y]`` input, also used
            to denormalize the output (AR scaling), or None to disable.
        transforms: list of transforms (train + valid); defaults to prediction_concat.
        augmentations: list of augmentation transforms (train only).
        aux_losses: list of auxiliary loss functions.
        grad_clip: max gradient norm for clipping, or None to disable; consider
            setting it in free-run mode, where long rollouts can spike gradients.
        plot_fn: plotting function for show_batch/show_results.
        device: target device (auto-detected if None).
        show_bar: whether to show tqdm progress bars.
        **kwargs: additional keyword arguments forwarded to ``NarxMLP``.
    """
    if train_mode not in ("free_run", "one_step"):
        raise ValueError(f"train_mode must be 'free_run' or 'one_step', got {train_mode!r}")
    if metrics is None:
        metrics = [fun_rmse]
    if transforms is None:
        transforms = [prediction_concat(t_offset=0)]
    if n_skip is None:
        n_skip = max(na, nb - 1)
    if washout is None:
        washout = n_skip

    inp, out = get_io_size(dls)
    model = NarxMLP(
        inp,
        out,
        na=na,
        nb=nb,
        hidden_size=hidden_size,
        num_layers=num_layers,
        teacher_forcing=train_mode == "one_step",
        washout=washout,
        **kwargs,
    )
    model = ScaledModel.from_dls(model, dls, input_norm, autoregressive=True)

    return Learner(
        model,
        dls,
        loss_func=loss_func,
        metrics=metrics,
        lr=lr,
        n_skip=n_skip,
        opt_func=opt_func,
        transforms=transforms,
        augmentations=augmentations,
        aux_losses=aux_losses,
        grad_clip=grad_clip,
        plot_fn=plot_fn,
        device=device,
        show_bar=show_bar,
    )


def TCNLearner(
    dls,
    num_layers: int = 3,
    hidden_size: int = 100,
    loss_func: nn.Module = nn.L1Loss(),
    metrics: list | None = None,
    lr: float = 3e-3,
    n_skip: int | None = None,
    opt_func: type = torch.optim.Adam,
    input_norm: type[Scaler] | None = StandardScaler,
    output_norm: type[Scaler] | None = None,
    transforms: list | None = None,
    augmentations: list | None = None,
    aux_losses: list | None = None,
    grad_clip: float | None = None,
    plot_fn=None,
    device: torch.device | None = None,
    show_bar: bool = True,
    **kwargs,
):
    """Create a Learner with a TCN model.

    Args:
        dls: DataLoaders providing training and validation data.
        num_layers: Number of TCN hidden layers (sets receptive field to 2**num_layers).
        hidden_size: Number of channels in hidden TCN layers.
        loss_func: Loss function instance.
        metrics: List of metric functions.
        lr: learning rate.
        n_skip: Number of initial time steps to skip in the loss (defaults to 2**num_layers).
        opt_func: Optimizer constructor.
        input_norm: Input normalization scaler class, or None to disable.
        output_norm: Output denormalization scaler class, or None to disable.
        transforms: list of transforms (train + valid).
        augmentations: list of augmentation transforms (train only).
        aux_losses: list of auxiliary loss functions.
        grad_clip: max gradient norm for clipping, or None to disable.
        plot_fn: plotting function for show_batch/show_results.
        device: target device (auto-detected if None).
        show_bar: whether to show tqdm progress bars.
        **kwargs: Additional arguments passed to ``TCN``.
    """
    if metrics is None:
        metrics = [fun_rmse]

    inp, out = get_io_size(dls)
    n_skip = 2**num_layers if n_skip is None else n_skip
    model = TCN(inp, out, num_layers, hidden_size, **kwargs)
    model = ScaledModel.from_dls(model, dls, input_norm, output_norm)

    return Learner(
        model,
        dls,
        loss_func=loss_func,
        metrics=metrics,
        lr=lr,
        n_skip=n_skip,
        opt_func=opt_func,
        transforms=transforms,
        augmentations=augmentations,
        aux_losses=aux_losses,
        grad_clip=grad_clip,
        plot_fn=plot_fn,
        device=device,
        show_bar=show_bar,
    )


def CRNNLearner(
    dls,
    loss_func: nn.Module = nn.L1Loss(),
    metrics: list | None = None,
    lr: float = 3e-3,
    n_skip: int = 0,
    opt_func: type = torch.optim.Adam,
    input_norm: type[Scaler] | None = StandardScaler,
    output_norm: type[Scaler] | None = None,
    transforms: list | None = None,
    augmentations: list | None = None,
    aux_losses: list | None = None,
    grad_clip: float | None = None,
    plot_fn=None,
    device: torch.device | None = None,
    show_bar: bool = True,
    **kwargs,
):
    """Create a Learner with a CRNN model.

    Args:
        dls: DataLoaders providing training and validation data.
        loss_func: Loss function instance.
        metrics: List of metric functions.
        lr: learning rate.
        n_skip: Number of initial time steps to skip in the loss.
        opt_func: Optimizer constructor.
        input_norm: Input normalization scaler class, or None to disable.
        output_norm: Output denormalization scaler class, or None to disable.
        transforms: list of transforms (train + valid).
        augmentations: list of augmentation transforms (train only).
        aux_losses: list of auxiliary loss functions.
        grad_clip: max gradient norm for clipping, or None to disable.
        plot_fn: plotting function for show_batch/show_results.
        device: target device (auto-detected if None).
        show_bar: whether to show tqdm progress bars.
        **kwargs: Additional arguments passed to ``CRNN``.
    """
    if metrics is None:
        metrics = [fun_rmse]

    inp, out = get_io_size(dls)
    model = CRNN(inp, out, **kwargs)
    model = ScaledModel.from_dls(model, dls, input_norm, output_norm)

    return Learner(
        model,
        dls,
        loss_func=loss_func,
        metrics=metrics,
        lr=lr,
        n_skip=n_skip,
        opt_func=opt_func,
        transforms=transforms,
        augmentations=augmentations,
        aux_losses=aux_losses,
        grad_clip=grad_clip,
        plot_fn=plot_fn,
        device=device,
        show_bar=show_bar,
    )


def AR_TCNLearner(
    dls,
    hl_depth: int = 3,
    alpha: float = 1,
    beta: float = 1,
    loss_func=nn.L1Loss(),
    metrics: list | None = None,
    lr: float = 3e-3,
    n_skip: int | None = None,
    opt_func: type = torch.optim.Adam,
    input_norm: type[Scaler] | None = StandardScaler,
    transforms: list | None = None,
    augmentations: list | None = None,
    aux_losses: list | None = None,
    grad_clip: float | None = None,
    plot_fn=None,
    device: torch.device | None = None,
    show_bar: bool = True,
    **kwargs,
):
    """Create a Learner with an autoregressive TCN model.

    Args:
        dls: DataLoaders providing training and validation data.
        hl_depth: Number of TCN hidden layers.
        alpha: Regularization weight for smoothness penalty.
        beta: Regularization weight for sparsity penalty.
        loss_func: loss function for training.
        metrics: Metric functions (defaults to RMSE).
        lr: learning rate.
        n_skip: Number of initial time steps to skip in the loss (defaults to 2**hl_depth).
        opt_func: Optimizer constructor.
        input_norm: Input normalization scaler class, or None to disable.
        transforms: list of transforms (train + valid); defaults to prediction_concat.
        augmentations: list of augmentation transforms (train only).
        aux_losses: list of auxiliary loss functions; defaults to activation regularizers.
        grad_clip: max gradient norm for clipping, or None to disable.
        plot_fn: plotting function for show_batch/show_results.
        device: target device (auto-detected if None).
        show_bar: whether to show tqdm progress bars.
        **kwargs: Additional arguments passed to ``TCN``.
    """
    if metrics is None:
        metrics = [fun_rmse]
    n_skip = 2**hl_depth if n_skip is None else n_skip

    if transforms is None:
        transforms = [prediction_concat(t_offset=0)]

    inp, out = get_io_size(dls)
    ar_model = AR_Model(TCN(inp + out, out, hl_depth, **kwargs), ar=False)

    if aux_losses is None:
        conv_module = ar_model.model.conv_layers[-1]
        aux_losses = [
            ActivationRegularizer(modules=[conv_module], alpha=alpha),
            TemporalActivationRegularizer(modules=[conv_module], beta=beta),
        ]

    model = ScaledModel.from_dls(ar_model, dls, input_norm, autoregressive=True)

    return Learner(
        model,
        dls,
        loss_func=loss_func,
        metrics=metrics,
        lr=lr,
        n_skip=n_skip,
        opt_func=opt_func,
        transforms=transforms,
        augmentations=augmentations,
        aux_losses=aux_losses,
        grad_clip=grad_clip,
        plot_fn=plot_fn,
        device=device,
        show_bar=show_bar,
    )
