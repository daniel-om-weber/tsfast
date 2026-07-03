"""NARX models: one-step regressors over explicit lag windows, simulated by output feedback."""

__all__ = [
    "NarxMLP",
    "NarxSpec",
]

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..cnn import CausalConv1d

_ACTS: dict[str, type[nn.Module]] = {
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "relu": nn.ReLU,
}


@dataclass(frozen=True)
class NarxSpec:
    """Static architecture description of the feedback path, used to generate backend kernels.

    The first layer receives the flattened lag buffer through the feedback weight (bias-free;
    the bias and the input-window contribution arrive precomputed per step), followed by the
    hidden layers and a linear output layer.

    Args:
        n_y: output dimension (also the width of one buffer slot)
        na: number of output lags in the buffer
        hidden: widths of the hidden layers (the lag-window layer is the first)
        act: activation name, a key of ``_ACTS``
    """

    n_y: int
    na: int
    hidden: tuple[int, ...]
    act: str

    @property
    def n_buf(self) -> int:
        """Flattened lag buffer size, ordered lag-major with the oldest sample first."""
        return self.na * self.n_y

    @property
    def dims(self) -> tuple[int, ...]:
        """Feature sizes through the feedback MLP: ``(n_buf, *hidden, n_y)``."""
        return (self.n_buf, *self.hidden, self.n_y)

    @property
    def n_linear(self) -> int:
        return len(self.hidden) + 1


class NarxMLP(nn.Module):
    """MLP NARX model ``y[t] = f(y[t-1..t-na], u[t..t-nb+1])``.

    The input sequence carries the concatenated ``[u, y]`` channels (as produced by the
    ``prediction_concat`` transform and by ``InferenceWrapper``); the split into input
    and output lags happens internally. The first hidden layer over the lag window is
    implemented as a pair of causal convolutions, so both execution paths share one set
    of weights:

    - **teacher forcing** (parallel): every timestep reads true outputs from the ``y``
      channel; the whole sequence evaluates as convolutions in one pass. This is the
      classic one-step-ahead NARX training mode.
    - **free run** (sequential): the ``y`` channel is read only for the first
      ``washout`` samples to fill the lag buffer; afterwards the model feeds back its
      own predictions and gradients flow through the feedback. This is simulation
      training, and the only mode that matches free-run evaluation.

    ``forward`` picks the path from the module state: teacher forcing is used only
    while ``self.training`` and ``teacher_forcing=True`` — in eval mode a NARX is
    always a simulator, which keeps validation losses and ``InferenceWrapper`` output
    free-run regardless of the training mode.

    The free-run rollout is irreducibly sequential in the fed-back outputs, so a naive
    per-step Python loop is dispatch-bound rather than FLOP-bound. The input-window
    contribution ``conv_u(u)`` is feedback-independent and always evaluates in parallel;
    only the feedback recurrence is dispatched to a backend:

    - ``"eager"``: plain Python loop — the reference implementation, any device and dtype.
    - ``"compiled"``: ``torch.compile`` over the loop — any device, slow first call.
    - ``"c"``: generated C++ recurrence with a fused BPTT backward, batch-parallel —
      float32 on CPU.
    - ``"triton"``: persistent-GEMV recurrence kernel with a fused BPTT backward —
      float32 on CUDA, padded widths up to 128.
    - ``"auto"``: ``triton`` when it applies, else ``compiled`` on CUDA; ``eager`` on
      CPU (select ``"c"`` explicitly to trade a one-time compilation for much faster
      CPU rollouts).

    All backends share the same parameters, so the backend can be switched at any time
    via the ``backend`` attribute. The fused backends are loss-agnostic
    ``autograd.Function``s and return the gradient w.r.t. the precomputed input-window
    contribution, so the convolutions stay inside plain autograd.

    Args:
        u_size: number of input channels ``u``.
        y_size: number of output channels ``y``.
        na: output lags per channel (autoregressive order).
        nb: input lags per channel, including the current sample (exogenous order).
        hidden_size: width of the hidden layers.
        num_layers: number of hidden layers (the lag-window layer is the first).
        act: activation name, one of ``tanh``, ``sigmoid``, ``relu``; bounded
            activations damp error feedback in free run.
        teacher_forcing: train one-step-ahead instead of free-running.
        washout: initial samples whose true outputs seed the lag buffer in free run.
        backend: execution backend of the free-run recurrence, see above.
    """

    def __init__(
        self,
        u_size: int,
        y_size: int,
        na: int = 8,
        nb: int = 8,
        hidden_size: int = 64,
        num_layers: int = 2,
        act: str = "tanh",
        teacher_forcing: bool = False,
        washout: int | None = None,
        backend: str = "auto",
    ):
        super().__init__()
        if na < 1 or nb < 1:
            raise ValueError(f"lag orders must be >= 1, got na={na}, nb={nb}")
        if act not in _ACTS:
            raise ValueError(f"unknown activation {act!r}, expected one of {sorted(_ACTS)}")
        self.u_size = u_size
        self.y_size = y_size
        self.na = na
        self.nb = nb
        self.teacher_forcing = teacher_forcing
        self.washout = max(na, nb - 1) if washout is None else washout
        self.backend = backend
        self.spec = NarxSpec(y_size, na, (hidden_size,) * num_layers, act)

        self.conv_u = CausalConv1d(u_size, hidden_size, nb)
        self.conv_y = nn.Conv1d(y_size, hidden_size, na, bias=False)
        self.act = _ACTS[act]()
        head = []
        for _ in range(num_layers - 1):
            head += [nn.Conv1d(hidden_size, hidden_size, 1), _ACTS[act]()]
        head.append(nn.Conv1d(hidden_size, y_size, 1))
        self.head = nn.Sequential(*head)
        self._compiled_rollout = None

    def _split(self, x: Tensor) -> tuple[Tensor, Tensor]:
        "Split the concatenated ``[u, y]`` sequence into channel-first tensors."
        if x.shape[-1] != self.u_size + self.y_size:
            raise ValueError(f"expected {self.u_size + self.y_size} input channels ([u, y]), got {x.shape[-1]}")
        return x[..., : self.u_size].transpose(1, 2), x[..., self.u_size :].transpose(1, 2)

    def _params_flat(self) -> list[Tensor]:
        """Feedback-path parameters in backend order: flat feedback weight, then head layers.

        The feedback weight is the ``conv_y`` kernel reshaped to act on the flattened
        lag-major buffer; the reshape is differentiable, so backend gradients flow back
        to the convolution parameters through autograd.
        """
        wy = self.conv_y.weight.permute(0, 2, 1).reshape(self.conv_y.weight.shape[0], self.spec.n_buf)
        params = [wy]
        for m in self.head:
            if isinstance(m, nn.Conv1d):
                params += [m.weight.squeeze(-1), m.bias]
        return params

    def _forward_teacher_forced(self, x: Tensor) -> Tensor:
        u, y = self._split(x)
        y_lags = F.pad(y, (self.na, 0))[..., :-1]  # window t-na .. t-1
        h = self.act(self.conv_u(u) + self.conv_y(y_lags))
        return self.head(h).transpose(1, 2)

    def _forward_free_run(self, x: Tensor) -> Tensor:
        u, y = self._split(x)
        hu = self.conv_u(u).transpose(1, 2).contiguous()  # feedback-independent, parallel
        y_true = y.transpose(1, 2).contiguous()
        match self._resolve_backend(x):
            case "eager":
                return self._rollout_eager(hu, y_true)
            case "compiled":
                if self._compiled_rollout is None:
                    torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 64)
                    self._compiled_rollout = torch.compile(self._rollout_eager, dynamic=False)
                return self._compiled_rollout(hu, y_true)
            case "c":
                from .backend_c import c_rollout

                return c_rollout(self.spec, hu, y_true, self.washout, self._params_flat())
            case "triton":
                from .backend_triton import triton_rollout

                return triton_rollout(self.spec, hu, y_true, self.washout, self._params_flat())
            case unknown:
                raise ValueError(f"unknown backend {unknown!r}")

    def _resolve_backend(self, x: Tensor) -> str:
        if self.backend != "auto":
            return self.backend
        if x.is_cuda and x.dtype == torch.float32:
            from . import backend_triton

            if backend_triton.is_available() and backend_triton.fits(self.spec):
                return "triton"
            return "compiled"
        return "eager"

    def _rollout_eager(self, hu: Tensor, y_true: Tensor) -> Tensor:
        w_y = self.conv_y.weight  # (hidden, y_size, na); index na-1 is lag 1
        history = [hu.new_zeros(hu.shape[0], self.y_size)] * self.na
        outputs = []
        for t in range(hu.shape[1]):
            y_lags = torch.stack(history[-self.na :], dim=-1)
            h = self.act(hu[:, t] + torch.einsum("hck,bck->bh", w_y, y_lags))
            y_t = self.head(h.unsqueeze(-1)).squeeze(-1)
            outputs.append(y_t)
            history.append(y_true[:, t] if t < self.washout else y_t)
        return torch.stack(outputs, dim=1)

    def forward(self, x: Tensor, ar: bool | None = None) -> Tensor:
        if ar is None:
            ar = not (self.training and self.teacher_forcing)
        return self._forward_free_run(x) if ar else self._forward_teacher_forced(x)


def check_rollout_args(spec: NarxSpec, hu: torch.Tensor, y_true: torch.Tensor, device_type: str) -> None:
    """Validate inputs of the fused backends, which are float32-only and device-specific."""
    if hu.device.type != device_type:
        raise RuntimeError(f"this backend requires {device_type} tensors, got {hu.device.type}")
    if hu.dtype != torch.float32 or y_true.dtype != torch.float32:
        raise RuntimeError(f"this backend requires float32, got {hu.dtype}")
    if hu.dim() != 3 or hu.shape[-1] != spec.hidden[0]:
        raise RuntimeError(f"expected hu of shape [B, L, {spec.hidden[0]}], got {tuple(hu.shape)}")
    if y_true.shape != (*hu.shape[:2], spec.n_y):
        raise RuntimeError(f"expected y_true of shape [{hu.shape[0]}, {hu.shape[1]}, {spec.n_y}]")


def fed_buffers(spec: NarxSpec, y_true: torch.Tensor, out: torch.Tensor, washout: int) -> torch.Tensor:
    """Reconstruct the flattened lag buffer fed into every step, ``[B*L, n_buf]``.

    The buffer contents are fully determined by the fed sequence (true outputs during
    washout, predictions after), so the training forward does not need to store them.
    """
    L = y_true.shape[1]
    t = torch.arange(L, device=y_true.device)
    fed = torch.where((t < washout)[None, :, None], y_true, out)
    padded = F.pad(fed, (0, 0, spec.na, 0))[:, :-1]  # window t-na .. t-1
    windows = padded.unfold(1, spec.na, 1)  # (B, L, n_y, na)
    return windows.permute(0, 1, 3, 2).reshape(-1, spec.n_buf)


def narx_param_grads(
    spec: NarxSpec,
    y_true: torch.Tensor,
    out: torch.Tensor,
    washout: int,
    zs: list[torch.Tensor],
    gy: torch.Tensor,
    gas: list[torch.Tensor],
) -> list[torch.Tensor]:
    """Parameter gradients of the rollout as batched GEMMs over the flattened adjoints.

    The buffer-adjoint recurrence is the only sequential part of BPTT; the parameter
    gradients are plain reductions over all ``B*L`` step samples. Shared by the C and
    Triton backends.

    Args:
        y_true: true output sequence ``[B, L, n_y]`` (read during washout).
        out: forward result ``[B, L, n_y]``.
        washout: teacher-forced prefix length.
        zs: stored post-activation hidden sequences, one ``[B, L, h]`` per hidden layer.
        gy: total adjoint of each step output ``[B, L, n_y]``.
        gas: pre-activation adjoints of the hidden layers, one ``[B, L, h]`` per layer.

    Returns:
        ``[dWy, dW_1, db_1, ..., dW_out, db_out]`` in ``_params_flat`` order.
    """
    acts = [fed_buffers(spec, y_true, out, washout)] + [z.reshape(-1, z.shape[-1]) for z in zs]
    adjoints = [ga.reshape(-1, ga.shape[-1]) for ga in gas] + [gy.reshape(-1, spec.n_y)]
    grads: list[torch.Tensor] = [adjoints[0].t() @ acts[0]]  # feedback weight has no bias
    for a_prev, ga in zip(acts[1:], adjoints[1:]):
        grads.append(ga.t() @ a_prev)
        grads.append(ga.sum(0))
    return grads
