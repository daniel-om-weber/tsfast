"""Physics-Informed RNN models with dual encoder architecture."""

__all__ = ["PIRNN", "AuxiliaryOutputLoss", "PIRNNLearner"]

import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections.abc import Callable

from ..training import Learner, fun_rmse, prediction_concat, truncate_sequence
from ..models.layers import SeqLinear
from ..models.scaling import ScaledModel, StandardScaler
from ..models.state import discover_state_spec, unflatten_state
from ..models.rnn import RNN
from ..prediction.fransys import Diag_RNN


class PIRNN(nn.Module):
    """Physics-Informed RNN with dual encoders: Sequence and State.

    Uses a diagnosis model (sequence encoder) to estimate initial hidden state
    from an initialization window, then a prognosis model to predict forward.
    Alternatively, an MLP state encoder maps a single physical state to hidden
    state for faster initialization.

    Args:
        n_u: Number of inputs.
        n_y: Number of outputs (total: supervised + auxiliary).
        init_sz: Initialization sequence length.
        prognosis: stateful model returning ``(output, state)``
        n_y_supervised: Number of supervised outputs (in dataset). Defaults to n_y.
        n_x: Number of extra states.
        hidden_size: Hidden state size (for default prognosis and diagnosis).
        rnn_layer: Number of RNN layers (for default prognosis and diagnosis).
        state_encoder_hidden: Hidden size for state encoder MLP.
        linear_layer: Linear layers in diagnosis RNN.
        final_layer: Final layer complexity.
        init_diag_only: Limit diagnosis to init_sz.
        default_encoder_mode: Default encoder mode during inference.
        p_state_encoder: Probability of using the state encoder per training batch.
        init_sz_range: If set, randomize ``init_sz`` uniformly within ``(min, max)``.
        **kwargs: Additional arguments passed to default diagnosis RNN.
    """

    def __init__(
        self,
        n_u: int,
        n_y: int,
        init_sz: int,
        prognosis: nn.Module | None = None,
        n_y_supervised: int | None = None,
        n_x: int = 0,
        hidden_size: int = 100,
        rnn_layer: int = 1,
        state_encoder_hidden: int = 64,
        linear_layer: int = 1,
        final_layer: int = 0,
        init_diag_only: bool = False,
        default_encoder_mode: str = "sequence",
        p_state_encoder: float = 0.0,
        init_sz_range: tuple[int, int] | None = None,
        **kwargs,
    ):
        super().__init__()
        n_y_supervised = n_y_supervised if n_y_supervised is not None else n_y
        self.n_u = n_u
        self.n_y = n_y
        self.n_x = n_x
        self.n_y_supervised = n_y_supervised
        self.init_sz = init_sz
        self.init_diag_only = init_diag_only
        self.hidden_size = hidden_size
        self.rnn_layer = rnn_layer
        self.default_encoder_mode = default_encoder_mode
        self.p_state_encoder = p_state_encoder
        self.init_sz_range = init_sz_range

        if prognosis is None:
            rnn_kwargs = dict(hidden_size=hidden_size, num_layers=rnn_layer, ret_full_hidden=True)
            rnn_kwargs = dict(rnn_kwargs, **kwargs)
            self.prognosis = RNN(n_u, **rnn_kwargs)
        else:
            self.prognosis = prognosis

        self._state_spec = discover_state_spec(self.prognosis, n_u, device="cpu")

        # Diagnosis RNN uses supervised outputs only
        self.diagnosis = Diag_RNN(
            n_u + n_x + n_y_supervised,
            self._state_spec.state_size,
            hidden_size=hidden_size,
            rnn_layer=rnn_layer,
            linear_layer=linear_layer,
            **kwargs,
        )

        # Auto-discover prognosis output feature dim
        with torch.no_grad():
            output, _ = self.prognosis(torch.zeros(2, 1, n_u))
        out_features = output.shape[-1]
        self.final = SeqLinear(out_features, n_y, hidden_layer=final_layer)

        # State encoder: physical state -> flat hidden state
        self.state_encoder = nn.Sequential(
            nn.Linear(n_y_supervised, state_encoder_hidden),
            nn.ReLU(),
            nn.Linear(state_encoder_hidden, self._state_spec.state_size),
        )

    def forward(
        self,
        x: torch.Tensor,
        init_state: list | None = None,
        encoder_mode: str = "default",
    ) -> torch.Tensor:
        """Forward pass with encoder mode auto-detection or explicit selection.

        Args:
            x: Input tensor [batch, seq, features].
            init_state: Initial hidden state. If None, estimated by encoder.
            encoder_mode: Encoder selection - 'none', 'sequence', or 'state'.
        """

        init_sz = random.randint(*self.init_sz_range) if self.training and self.init_sz_range else self.init_sz
        self._effective_init_sz = init_sz

        u = x[:, :, : self.n_u]
        # Use n_y_supervised for initialization sequence (only supervised outputs in data)
        x_init = x[:, :init_sz, : self.n_u + self.n_x + self.n_y_supervised]
        if encoder_mode == "default":
            if self.training and self.p_state_encoder > 0:
                encoder_mode = "state" if random.random() < self.p_state_encoder else "sequence"
            else:
                encoder_mode = self.default_encoder_mode

        if encoder_mode == "none":
            return self._forward_predictor(u, init_state)
        elif encoder_mode == "sequence":
            return self._forward_sequence_encoder(u[:, init_sz:], x_init, init_state)
        elif encoder_mode == "state":
            return self._forward_state_encoder(u[:, init_sz:], x_init, init_state)
        else:
            raise ValueError(f"encoder_mode must be 'none', 'sequence', or 'state', got {encoder_mode}")

    @staticmethod
    def _diag_output(result) -> torch.Tensor:
        """Extract flat diagnosis tensor from model output (handles tuple or tensor)."""
        return result[0] if isinstance(result, tuple) else result

    def _forward_sequence_encoder(
        self,
        u: torch.Tensor,
        x_init: torch.Tensor,
        init_state: list = None,
    ) -> torch.Tensor:
        """Forward using sequence encoder (diagnosis RNN).

        Args:
            u: Prognosis input [batch, seq - init_sz, n_u].
            x_init: Initialization window [batch, init_sz, n_u + n_x + n_y_supervised].
            init_state: Initial hidden state. If None, estimated from x_init.

        Returns:
            Predictions [batch, seq, n_y] covering both init and prognosis windows.
        """
        out_init = self._diag_output(self.diagnosis(x_init))
        if init_state is None:
            init_state = unflatten_state(out_init[:, -1], self._state_spec)
        out_prog, self.new_hidden = self.prognosis(u, init_state)
        if out_prog.dim() > 3:
            out_prog = out_prog[-1]
        result = self.final(out_prog)
        return F.pad(result, (0, 0, self._effective_init_sz, 0))

    def _forward_state_encoder(
        self,
        u: torch.Tensor,
        x_init: torch.Tensor,
        init_state,
    ) -> torch.Tensor:
        """Forward using state encoder (MLP), zero-pads init window.

        Args:
            u: Prognosis input [batch, seq - init_sz, n_u].
            x_init: Init window [batch, init_sz, n_x + n_y_supervised].
            init_state: Pre-computed hidden state, or None to encode from x_init last step.

        Returns:
            Predictions [batch, seq, n_y] with init window zero-padded.
        """
        if init_state is None:  # If init_state is not provided, use last initialization step
            init_state = x_init[:, -1, -self.n_y_supervised :]
        init_state = unflatten_state(self.encode_single_state(init_state), self._state_spec)
        pred = self._forward_predictor(u, init_state)
        return F.pad(pred, (0, 0, self._effective_init_sz, 0))  # Zero-pad init window

    def _forward_predictor(
        self,
        u: torch.Tensor,
        init_state: list,
    ) -> torch.Tensor:
        """Forward using predictor model.

        Args:
            u: Input tensor [batch, seq, n_u].
            init_state: Initial hidden state.

        Returns:
            Predictions [batch, seq, n_y].
        """
        out_prog, _ = self.prognosis(u, init_state)
        if out_prog.dim() > 3:
            out_prog = out_prog[-1]
        return self.final(out_prog)

    def encode_single_state(self, physical_state: torch.Tensor) -> torch.Tensor:
        """Convert single physical state to flat hidden state vector.

        Args:
            physical_state: Physical state ``[batch, n_y_supervised]``.

        Returns:
            Flat hidden state ``[batch, state_size]``.
        """
        return self.state_encoder(physical_state)


class AuxiliaryOutputLoss:
    """Wrapper that applies loss only to supervised outputs, ignoring auxiliary outputs.

    Args:
        loss_func: Loss function to wrap.
        n_supervised: Number of supervised output channels.
    """

    def __init__(
        self,
        loss_func: Callable,
        n_supervised: int,
    ):
        self.loss_func = loss_func
        self.n_supervised = n_supervised

    def __call__(
        self,
        pred: torch.Tensor,
        targ: torch.Tensor,
    ) -> torch.Tensor:
        """Apply loss only to first n_supervised channels of predictions."""
        return self.loss_func(pred[..., : self.n_supervised], targ)


def PIRNNLearner(
    dls,
    init_sz: int,
    n_aux_outputs: int = 0,
    attach_output: bool = False,
    loss_func: Callable = nn.L1Loss(),
    metrics: list | None = None,
    opt_func: Callable = torch.optim.Adam,
    lr: float = 3e-3,
    transforms: list | None = None,
    augmentations: list | None = None,
    aux_losses: list | None = None,
    input_norm: type | None = StandardScaler,
    output_norm: type | None = None,
    prognosis: nn.Module | None = None,
    hidden_size: int = 100,
    rnn_layer: int = 1,
    **kwargs,
) -> Learner:
    """Create PIRNN learner with appropriate configuration.

    Args:
        dls: DataLoaders.
        init_sz: Initialization sequence length.
        n_aux_outputs: Number of auxiliary outputs (not in dataset).
        attach_output: Whether to attach output to input via prediction_concat.
        loss_func: Loss function.
        metrics: Metrics.
        opt_func: Optimizer.
        lr: Learning rate.
        transforms: Additional transforms (train + valid).
        augmentations: Additional augmentations (train only).
        aux_losses: Additional auxiliary losses.
        input_norm: Input normalization Scaler class.
        output_norm: Output denormalization Scaler class.
        prognosis: Custom prognosis model (default: RNN with ret_full_hidden=True).
        hidden_size: Hidden units for default prognosis and diagnosis.
        rnn_layer: Number of RNN layers for default prognosis and diagnosis.
        **kwargs: Additional arguments for PIRNN.
    """
    if metrics is None:
        metrics = [fun_rmse]
    transforms = list(transforms) if transforms else []
    augmentations = list(augmentations) if augmentations else []
    aux_losses = list(aux_losses) if aux_losses else []

    _batch = dls.one_batch()
    inp = _batch[0].shape[-1]
    out = _batch[1].shape[-1]  # Supervised outputs from dataset
    n_y_total = out + n_aux_outputs  # Total outputs (supervised + auxiliary)

    norm_u, norm_y = dls.norm_stats

    if attach_output:
        n_u = inp
        if not any(isinstance(t, prediction_concat) for t in transforms):
            transforms.insert(0, prediction_concat(t_offset=0))
        combined_input_stats = norm_u + norm_y
    else:
        n_u = inp - out
        combined_input_stats = norm_u + norm_y

    if prognosis is None:
        rnn_kwargs_inner = {
            k: v
            for k, v in kwargs.items()
            if k
            not in {
                "n_y_supervised",
                "n_x",
                "state_encoder_hidden",
                "linear_layer",
                "final_layer",
                "init_diag_only",
                "default_encoder_mode",
                "p_state_encoder",
                "init_sz_range",
                "diag_model",
            }
        }
        prognosis = RNN(n_u, hidden_size=hidden_size, num_layers=rnn_layer, ret_full_hidden=True, **rnn_kwargs_inner)

    model = PIRNN(
        n_u,
        n_y_total,
        init_sz,
        prognosis=prognosis,
        n_y_supervised=out,
        hidden_size=hidden_size,
        rnn_layer=rnn_layer,
        **kwargs,
    )

    # Wrap model with input normalization and optional output denormalization
    if input_norm is not None:
        in_scaler = input_norm.from_stats(combined_input_stats)
        out_scaler = output_norm.from_stats(norm_y) if output_norm is not None else None
        model = ScaledModel(model, in_scaler, out_scaler)

    # For long sequences, add truncate_sequence augmentation
    seq_len = _batch[0].shape[1]
    LENGTH_THRESHOLD = 300
    if seq_len > init_sz + LENGTH_THRESHOLD:
        if not any(isinstance(a, truncate_sequence) for a in augmentations):
            INITIAL_SEQ_LEN = 100
            augmentations.append(truncate_sequence(init_sz + INITIAL_SEQ_LEN))

    # Wrap loss and metrics to only use supervised outputs when auxiliary outputs present
    if n_aux_outputs > 0:
        loss_func = AuxiliaryOutputLoss(loss_func, out)
        metrics = [AuxiliaryOutputLoss(m, out) for m in metrics]

    return Learner(
        model,
        dls,
        loss_func=loss_func,
        metrics=metrics,
        n_skip=init_sz,
        opt_func=opt_func,
        lr=lr,
        transforms=transforms,
        augmentations=augmentations,
        aux_losses=aux_losses,
    )
