"""Damped-ansatz DD-PINN surrogate: continuous-time model, scaler, and physics-only trainer."""

__all__ = [
    "DampedAnsatzPINN",
    "DDPINNRollout",
    "make_collocation_dls",
    "SurrogatePINNLearner",
]

from collections.abc import Callable

import torch
from torch import Tensor, nn

from ..models.layers import SeqLinear
from ..models.scaling import MinMaxScaler
from ..training.learner import Learner
from ..tsdata.pipeline import DataLoaders


def _bounds(ranges: list[tuple[float, float]]) -> tuple[Tensor, Tensor]:
    """Split ``[(lo, hi), ...]`` modelling ranges into lower/upper bound tensors."""
    lo = torch.tensor([r[0] for r in ranges], dtype=torch.float32)
    hi = torch.tensor([r[1] for r in ranges], dtype=torch.float32)
    return lo, hi


class DampedAnsatzPINN(nn.Module):
    """Continuous-time surrogate with an analytically-differentiable damped ansatz.

    Operates entirely in normalized ``[-1, 1]`` coordinates. The input row is laid out as
    ``[x_k (n_state) | cond (n_cond) | t (1)]`` where the last channel is normalized time.
    The MLP trunk maps ``(x_k, cond)`` to ``4·n_ansatz·n_state`` coefficients — time is
    decoupled from the trunk and enters only through the closed form. Per output channel,
    summed over ``k = 1..n_ansatz`` with ansatz-time ``τ = t − t_zero``::

        g_k(τ) = a_k·(sin(b_k·τ + c_k)·exp(−d_k·τ) − sin(c_k))
        x(τ)   = x_k + Σ_k g_k(τ)

    ``g_k(0) = 0`` makes ``x(0) = x_k`` exact (the initial condition is enforced by
    construction — no IC loss), and ``dx/dτ`` is closed-form, so the physics residual needs
    no autograd over time.

    Args:
        n_state: state dimension; ``x_k`` occupies the first ``n_state`` input channels and
            equals the output dimension.
        n_cond: conditioning channels (control + collocation vars) between ``x_k`` and ``t``.
        n_ansatz: number of damped ansatz functions summed per output channel.
        hidden_size: width of the trunk MLP.
        hidden_layer: number of hidden layers in the trunk MLP.
        act: trunk activation function class.
        t_zero: normalized-time value mapped to ansatz-time ``τ = 0`` (raw ``t = 0``); ``-1``
            for a ``[-1, 1]`` time box.
        exp_clamp: upper bound on the decay exponent ``−d·τ``, guarding float32 overflow.
            The trunk emits ``d`` unconstrained (as in the reference implementation), so it may be
            negative. **Anti-damping failure mode:** when ``d < 0`` the exponent ``−d·τ`` turns
            positive and ``exp(−d·τ)`` *grows* with ``τ`` instead of decaying — the basis function
            blows up over the window rather than damping. This clamp only caps the exponent to
            avoid overflow; it does not suppress that growth, and wherever it fires the analytic
            ``dx/dτ`` (from the *unclamped* closed form) diverges from autograd through the clamp.
            In the normal damped regime (``d ≥ 0`` ⇒ exponent ``≤ 0``) it never fires and the
            analytic derivative stays exact. Persistent instability in long autoregressive rollouts
            is the symptom of learned ``d < 0``; the remedy — constraining ``d`` non-negative (e.g.
            ``softplus``) — is deliberately omitted to match the unconstrained reference behavior.
    """

    def __init__(
        self,
        n_state: int,
        n_cond: int,
        n_ansatz: int = 50,
        hidden_size: int = 64,
        hidden_layer: int = 3,
        act: type = nn.Tanh,
        t_zero: float = -1.0,
        exp_clamp: float = 30.0,
    ):
        super().__init__()
        self.n_state = n_state
        self.n_cond = n_cond
        self.n_ansatz = n_ansatz
        self.t_zero = t_zero
        self.exp_clamp = exp_clamp
        self.trunk = SeqLinear(
            n_state + n_cond,
            4 * n_ansatz * n_state,
            hidden_size=hidden_size,
            hidden_layer=hidden_layer,
            act=act,
        )

    def forward(self, x: Tensor, derivative_flag: bool = False):
        B, S = x.shape[0], x.shape[1]
        feat = x[..., :-1]  # (x_k, cond) — time stripped
        x_k = x[..., : self.n_state]
        tau = (x[..., -1:] - self.t_zero).unsqueeze(-1)  # [B, S, 1, 1]
        coeffs = self.trunk(feat).reshape(B, S, self.n_state, self.n_ansatz, 4)
        a, b, c, d = coeffs.unbind(-1)  # each [B, S, n_state, n_ansatz]
        phase = b * tau + c
        decay = torch.exp((-d * tau).clamp(max=self.exp_clamp))
        y = x_k + (a * (torch.sin(phase) * decay - torch.sin(c))).sum(-1)
        if not derivative_flag:
            return y
        dy_dtau = (a * (b * torch.cos(phase) - d * torch.sin(phase)) * decay).sum(-1)
        return y, dy_dtau


class DDPINNRollout(nn.Module):
    """Autoregressive rollout of a :class:`DampedAnsatzPINN` as a differentiable sequence model.

    Turns the pointwise continuous-time map into a sequence-to-sequence model: given a physical
    initial state and a physical conditioning sequence, it predicts the physical state trajectory
    by feeding each one-step prediction back as the next initial state. The loop runs in normalized
    ``[-1, 1]`` space (the model's native coordinates); the two scalers it carries bridge to
    physical units at the boundary, so callers work entirely in physical units. ``forward`` is
    differentiable end-to-end — use it inside a training step for a multi-step / data-assisted loss,
    or under :func:`torch.no_grad` for inference. :meth:`step` is the atomic normalized-space unit
    the loop iterates (and the natural single-step target for ONNX export).

    Args:
        model: a trained :class:`DampedAnsatzPINN` (called without the derivative at inference).
        state_scaler: the :class:`~tsfast.models.scaling.MinMaxScaler` for the state channels.
        cond_scaler: the :class:`~tsfast.models.scaling.MinMaxScaler` for the conditioning channels (may be empty).
        t_sample: physical step size per autoregressive step; must be ``<= t_max``.
        t_max: physical horizon defining the ``[0, t_max] -> [-1, 1]`` time map.
    """

    def __init__(
        self,
        model: nn.Module,
        state_scaler: nn.Module,
        cond_scaler: nn.Module,
        t_sample: float,
        t_max: float,
    ):
        super().__init__()
        if t_sample > t_max:
            raise ValueError(
                f"t_sample ({t_sample}) must be <= t_max ({t_max}); the surrogate is only trained on t in [0, t_max]."
            )
        self.model = model
        self.state_scaler = state_scaler
        self.cond_scaler = cond_scaler
        # physical t_sample mapped into the [0, t_max] -> [-1, 1] time box
        self.register_buffer("t_scaled", torch.tensor((2.0 / t_max) * t_sample - 1.0).view(1, 1, 1))

    def step(self, xk_norm: Tensor, cond_norm: Tensor) -> Tensor:
        """One autoregressive step in normalized coordinates: ``[B, 1, n_state] -> [B, 1, n_state]``."""
        t = self.t_scaled.expand(xk_norm.shape[0], 1, 1)
        return self.model(torch.cat([xk_norm, cond_norm, t], dim=-1))

    def forward(self, x0: Tensor, cond_seq: Tensor) -> Tensor:
        """Roll out ``N`` steps in physical units.

        Args:
            x0: physical initial state ``[B, n_state]``.
            cond_seq: physical conditioning sequence ``[B, N, n_cond]``; ``N`` sets the horizon and
                the width may be 0 when there are no conditioning channels.

        Returns:
            Physical-unit state trajectory ``[B, N, n_state]``.
        """
        xk = self.state_scaler.normalize(x0.unsqueeze(1))  # [B, 1, n_state]
        cond = self.cond_scaler.normalize(cond_seq)  # [B, N, n_cond]
        outs = []
        for k in range(cond.shape[1]):
            xk = self.step(xk, cond[:, k : k + 1])
            outs.append(xk)
        return self.state_scaler.denormalize(torch.cat(outs, dim=1))


class _CollocationStream(torch.utils.data.IterableDataset):
    """Finite stream of freshly-sampled collocation batches (re-drawn each epoch)."""

    def __init__(self, gen_fn: Callable, bs: int, seq_len: int, n_batches: int):
        self.gen_fn, self.bs, self.seq_len, self.n_batches = gen_fn, bs, seq_len, n_batches

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        for _ in range(self.n_batches):
            X = self.gen_fn(self.bs, self.seq_len, "cpu")
            yield X, X.new_zeros(X.shape[0], 1, 1)


class _CachedBatches(torch.utils.data.IterableDataset):
    """Fixed pre-generated batches, for a validation metric that is stable across epochs."""

    def __init__(self, batches: list[Tensor]):
        self.batches = batches

    def __len__(self):
        return len(self.batches)

    def __iter__(self):
        for X in self.batches:
            yield X, X.new_zeros(X.shape[0], 1, 1)


def make_collocation_dls(
    generate_pinn_input: Callable,
    bs: int,
    steps_per_epoch: int,
    val_steps: int,
    val_seed: int = 0,
    seq_len: int = 1,
) -> DataLoaders:
    """Build a :class:`DataLoaders` of collocation batches for physics-only training.

    The training stream re-samples fresh collocation points every epoch; the validation set
    is generated once under a fixed seed so its physics metric is comparable across epochs.
    ``len(dls.train) == steps_per_epoch`` so the scheduler receives the right total-step
    count. Batches are ``(X, dummy_y)`` pairs — the dummy target is ignored by the
    physics-only learner.
    """
    train_ds = _CollocationStream(generate_pinn_input, bs, seq_len, steps_per_epoch)
    # Fork/seed only the CPU generator: touching the CUDA generators (e.g. via torch.manual_seed)
    # would initialize a CUDA context on every visible GPU, even in CPU-only processes.
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(val_seed)
        val_batches = [generate_pinn_input(bs, seq_len, "cpu") for _ in range(val_steps)]
    valid_ds = _CachedBatches(val_batches)
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=None, num_workers=0)
    valid_dl = torch.utils.data.DataLoader(valid_ds, batch_size=None, num_workers=0)
    return DataLoaders(train_dl, valid_dl)


def _zero_loss(pred: Tensor, targ: Tensor) -> Tensor:
    """Placeholder primary loss; the physics-only learner overrides training and validation."""
    return torch.zeros((), device=pred.device)


class SurrogatePINNLearner(Learner):
    """Physics-only (no-data) trainer for a :class:`DampedAnsatzPINN`.

    Each step samples a fresh batch of collocation points (in normalized ``[-1, 1]``
    coordinates) and minimizes the ODE residual built from the ansatz's analytic ``dx/dt``.
    Normalization and the chain-rule factor live here, so the model stays in normalized
    space and the user's ``residual_func`` only ever sees physical quantities. Reuses
    :class:`~tsfast.training.learner.Learner` for the optimizer, scheduler, NaN guard, and
    progress reporting.

    Args:
        model: a :class:`DampedAnsatzPINN` (or compatible ``(y, dy_dtau)`` model).
        generate_pinn_input: ``(bs, seq_len, device) -> Tensor`` collocation sampler in
            normalized ``[-1, 1]`` coordinates, row layout ``[x_k | cond | t]``.
        residual_func: ``(x_phys, cond_phys, dxdt_phys) -> scalar`` ODE residual, all in
            physical units.
        state_range: list of ``(lo, hi)`` physical bounds per state channel.
        cond_range: list of ``(lo, hi)`` physical bounds per conditioning channel (may be
            empty).
        t_max: physical time horizon mapped to the upper edge of the ``[-1, 1]`` time box.
        steps_per_epoch: collocation batches drawn per epoch (sets the scheduler length).
        bs: collocation batch size.
        val_steps: number of fixed validation batches.
        val_seed: seed for the deterministic validation set.
        lr: learning rate.
        metrics: optional metrics (unused by the physics-only ``validate``).
    """

    def __init__(
        self,
        model: nn.Module,
        generate_pinn_input: Callable,
        residual_func: Callable,
        state_range: list[tuple[float, float]],
        cond_range: list[tuple[float, float]],
        t_max: float,
        steps_per_epoch: int = 200,
        bs: int = 4096,
        val_steps: int = 20,
        val_seed: int = 0,
        lr: float = 3e-3,
        metrics: list | None = None,
        **kw,
    ):
        self.residual_func = residual_func
        self.state_scaler = MinMaxScaler(*_bounds(state_range), feature_range=(-1.0, 1.0))
        self.cond_scaler = MinMaxScaler(*_bounds(cond_range), feature_range=(-1.0, 1.0))
        self.n_state = len(state_range)
        self.n_cond = len(cond_range)
        self.t_max = t_max
        # chain-rule factor: time-slope / state-slope, converting scaled dy/dτ -> physical dx/dt
        self.deriv_factor = (2.0 / t_max) / self.state_scaler.scale  # [1, 1, n_state]
        dls = make_collocation_dls(generate_pinn_input, bs, steps_per_epoch, val_steps, val_seed)
        super().__init__(model, dls, loss_func=_zero_loss, lr=lr, metrics=metrics, **kw)

    def setup(self, lr: float | None = None, scheduler_fn: Callable | None = None, n_epoch: int | None = None):
        """Standard setup plus moving the scalers and chain-rule factor to the device."""
        super().setup(lr=lr, scheduler_fn=scheduler_fn, n_epoch=n_epoch)
        self.state_scaler.to(self.device)
        self.cond_scaler.to(self.device)
        self.deriv_factor = self.deriv_factor.to(self.device)

    def physics_loss(self, X: Tensor) -> Tensor:
        """ODE residual on a batch of normalized collocation points."""
        y, dy_dtau = self.model(X, derivative_flag=True)  # normalized
        x_phys = self.state_scaler.denormalize(y)
        dxdt_phys = dy_dtau * self.deriv_factor
        cond_norm = X[..., self.n_state : self.n_state + self.n_cond]
        cond_phys = self.cond_scaler.denormalize(cond_norm)
        return self.residual_func(x_phys, cond_phys, dxdt_phys)

    def training_step(self, xb: Tensor, yb: Tensor) -> float:
        loss = self.physics_loss(xb)
        if torch.isnan(loss):
            self.opt.zero_grad()
            return float("nan")
        self.backward_step(loss)
        return loss.item()

    def validate(self, dl=None, chunk_sz: int | None = None) -> tuple[float, dict[str, float]]:
        """Mean physics residual over the (fixed) validation collocation set.

        The derivative is analytic, so this runs entirely under ``no_grad``.
        """
        dl = dl or self.dls.valid
        self.model.eval()
        losses = []
        with torch.no_grad():
            for batch in dl:
                xb, _ = self.prepare_batch(batch, training=False)
                losses.append(self.physics_loss(xb).item())
        return sum(losses) / max(1, len(losses)), {}

    def as_rollout(self, t_sample: float | None = None) -> DDPINNRollout:
        """Bundle the trained model with both scalers into a differentiable sequence rollout.

        The returned :class:`DDPINNRollout` is a single self-contained ``nn.Module`` (physical state
        + physical conditioning sequence in, physical trajectory out) that ``torch.save`` persists as
        one deployable artifact. Call it under :func:`torch.no_grad` for inference, or inside a
        training step for a multi-step / data-assisted loss.

        Args:
            t_sample: physical step size per autoregressive step (defaults to ``t_max``).
        """
        t_sample = self.t_max if t_sample is None else t_sample
        return DDPINNRollout(self.model, self.state_scaler, self.cond_scaler, t_sample, self.t_max).to(self.device)
