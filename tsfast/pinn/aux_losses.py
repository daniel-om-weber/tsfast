"""PINN-specific loss callables for physics-informed training."""

__all__ = [
    "PhysicsLoss",
    "CollocationLoss",
    "ConsistencyLoss",
    "TransitionSmoothnessLoss",
]

from collections.abc import Callable

import torch
import torch.nn.functional as F

from ..models.state import unflatten_state
from .differentiation import diff2_forward
from .signals import generate_random_states


class PhysicsLoss:
    """Physics-informed loss term for PINN training.

    Args:
        physics_loss_func: function(u, y_pred, y_ref) returning dict of losses or single loss tensor
        weight: global scaling factor for physics loss contribution
        loss_weights: per-component weights like {'physics': 1.0, 'derivative': 0.1}
        n_inputs: number of input channels (if using concatenated inputs like FranSysLearner)
        n_skip: number of initial timesteps to skip before computing physics loss
    """

    def __init__(
        self,
        physics_loss_func: Callable,
        weight: float = 1.0,
        loss_weights: dict | None = None,
        n_inputs: int | None = None,
        n_skip: int = 0,
    ):
        self.physics_loss_func = physics_loss_func
        self.weight = weight
        self.loss_weights = loss_weights or {}
        self.n_inputs = n_inputs
        self.n_skip = n_skip

    def __call__(self, pred: torch.Tensor, yb: torch.Tensor, xb: torch.Tensor) -> torch.Tensor:
        """Compute physics-informed loss on training data."""
        # Extract input part (if n_inputs specified, split concatenated input)
        if self.n_inputs is not None:
            u = xb[:, :, : self.n_inputs]
        else:
            u = xb

        # Get predictions and ground truth (already in raw physical space)
        y_pred = pred
        y_ref = yb

        # Skip initial timesteps (e.g., init window for state encoder)
        if self.n_skip > 0:
            u = u[:, self.n_skip :]
            y_pred = y_pred[:, self.n_skip :]
            y_ref = y_ref[:, self.n_skip :]

        # Compute physics losses (system-specific)
        loss_dict = self.physics_loss_func(u, y_pred, y_ref)

        # Handle both dict and single tensor returns
        if isinstance(loss_dict, dict):
            physics_total = sum(self.loss_weights.get(k, 1.0) * v for k, v in loss_dict.items())
        else:
            physics_total = loss_dict

        return self.weight * physics_total


class _CollocationDataset(torch.utils.data.IterableDataset):
    """Module-level dataset for collocation point generation (picklable for spawn multiprocessing)."""

    def __init__(self, gen_fn, bs, seq_len):
        self.gen_fn = gen_fn
        self.bs = bs
        self.seq_len = seq_len

    def __iter__(self):
        while True:
            yield self.gen_fn(self.bs, self.seq_len, "cpu")


class CollocationLoss:
    """Collocation-point physics loss for PINN training.

    Args:
        generate_pinn_input: function(batch_size, seq_len, device) returning tensor of collocation points
        physics_loss_func: function(u, y_pred, y_ref) returning dict of losses or single loss tensor
        weight: global scaling factor for physics loss contribution
        loss_weights: per-component weights like {'physics': 1.0, 'derivative': 0.1}
        num_workers: number of parallel workers for collocation point generation
        init_mode: initialization mode: 'none', 'state_encoder', or 'random_hidden'
        output_ranges: list of (min, max) tuples for random state generation
        hidden_std: std for random hidden state initialization
        n_skip: number of initial timesteps to skip before computing physics loss
    """

    def __init__(
        self,
        generate_pinn_input: Callable,
        physics_loss_func: Callable,
        weight: float = 1.0,
        loss_weights: dict | None = None,
        num_workers: int = 0,
        init_mode: str = "none",
        output_ranges: list | None = None,
        hidden_std: float = 0.1,
        n_skip: int = 0,
    ):
        self.generate_pinn_input = generate_pinn_input
        self.physics_loss_func = physics_loss_func
        self.weight = weight
        self.loss_weights = loss_weights or {}
        self.num_workers = num_workers
        self.loader_iter = None
        self.init_mode = init_mode
        self.output_ranges = output_ranges
        self.hidden_std = hidden_std
        self.n_skip = n_skip
        self.model = None
        self.inner_model = None

    def setup(self, trainer):
        """Resolve model references from the trainer."""
        self.model = trainer.model
        from ..models.scaling import unwrap_model

        self.inner_model = unwrap_model(trainer.model)

    def teardown(self, trainer):
        """Clean up the collocation DataLoader iterator."""
        if self.loader_iter is not None:
            if hasattr(self.loader_iter, "_stop"):
                self.loader_iter._stop.set()
            elif hasattr(self.loader_iter, "_workers_done_event"):
                from ..tsdata.safe_iter import drain_worker_queue

                drain_worker_queue(self.loader_iter)
            self.loader_iter = None

    def _prepare_loader(self, u_real):
        """Create DataLoader for collocation point generation."""
        ds = _CollocationDataset(self.generate_pinn_input, u_real.shape[0], u_real.shape[1])
        if self.num_workers > 0:
            return torch.utils.data.DataLoader(
                ds,
                batch_size=None,
                num_workers=self.num_workers,
                prefetch_factor=2,
            )
        from ..tsdata.prefetch import PrefetchLoader

        dl = torch.utils.data.DataLoader(ds, batch_size=None, num_workers=0)
        return PrefetchLoader(dl, prefetch=2)

    def __call__(self, pred: torch.Tensor, yb: torch.Tensor, xb: torch.Tensor) -> torch.Tensor:
        """Compute physics-informed loss on collocation points."""
        device = xb.device
        batch_size = xb.shape[0]

        if self.loader_iter is None:
            self.loader_iter = iter(self._prepare_loader(xb))

        u_coloc = next(self.loader_iter).to(device)

        y_ref = None

        if self.init_mode == "state_encoder":
            if self.output_ranges is None:
                raise ValueError("output_ranges must be provided when init_mode='state_encoder'")

            n_outputs = len(self.output_ranges)
            physical_states = generate_random_states(batch_size, n_outputs, self.output_ranges, device)

            if hasattr(self.inner_model, "encode_single_state"):
                with torch.enable_grad():
                    y_pred = self.model(u_coloc, init_state=physical_states, encoder_mode="state")
                y_ref = physical_states.unsqueeze(1).expand(-1, u_coloc.shape[1], -1)
            else:
                raise ValueError("Model must have encode_single_state method for init_mode='state_encoder'")

        elif self.init_mode == "random_hidden":
            if hasattr(self.inner_model, "_state_spec"):
                state_size = self.inner_model._state_spec.state_size
                init_flat = torch.randn(batch_size, state_size, device=device) * self.hidden_std
                init_hidden = unflatten_state(init_flat, self.inner_model._state_spec)
                with torch.enable_grad():
                    y_pred = self.model(u_coloc, init_state=init_hidden, encoder_mode="none")
                y_ref = None
            else:
                raise ValueError("Model structure not compatible with init_mode='random_hidden'")

        else:
            with torch.enable_grad():
                y_pred = self.model(u_coloc)
            y_ref = None

        # Skip initial timesteps (e.g., init window for state encoder)
        if self.n_skip > 0:
            u_coloc = u_coloc[:, self.n_skip :]
            y_pred = y_pred[:, self.n_skip :]
            if y_ref is not None:
                y_ref = y_ref[:, self.n_skip :]

        loss_dict = self.physics_loss_func(u_coloc, y_pred, y_ref)

        if isinstance(loss_dict, dict):
            physics_total = sum(self.loss_weights.get(k, 1.0) * v for k, v in loss_dict.items())
        else:
            physics_total = loss_dict

        return self.weight * physics_total


class ConsistencyLoss:
    """Consistency loss between sequence and state encoders.

    Args:
        weight: weight for consistency loss
        match_at_timestep: timestep to match hidden states (default: model.init_sz)
    """

    def __init__(
        self,
        weight: float = 1.0,
        match_at_timestep: int | None = None,
    ):
        self.weight = weight
        self.match_at_timestep = match_at_timestep
        self._hook = None
        self._diag_output = None
        self.inner_model = None
        self._has_modules = False

    def _capture_diag(self, module, input, output):
        """Hook callback to capture diagnosis RNN output."""
        self._diag_output = output[0] if isinstance(output, tuple) else output

    def setup(self, trainer):
        """Register forward hook on diagnosis if model supports it."""
        from ..models.scaling import unwrap_model

        self.inner_model = unwrap_model(trainer.model)
        if hasattr(self.inner_model, "diagnosis") and hasattr(self.inner_model, "encode_single_state"):
            self._hook = self.inner_model.diagnosis.register_forward_hook(self._capture_diag)
            self._has_modules = True

    def teardown(self, trainer):
        """Remove the forward hook."""
        if self._hook is not None:
            self._hook.remove()
            self._hook = None
        self._has_modules = False

    def __call__(self, pred: torch.Tensor, yb: torch.Tensor, xb: torch.Tensor) -> torch.Tensor:
        """Compute consistency loss between SequenceEncoder and StateEncoder."""
        if not self._has_modules:
            return torch.tensor(0.0, device=pred.device)
        if self._diag_output is None:
            return torch.tensor(0.0, device=pred.device)

        timestep = self.match_at_timestep
        if timestep is None and hasattr(self.inner_model, "init_sz"):
            timestep = getattr(self.inner_model, "_effective_init_sz", self.inner_model.init_sz) - 1
        elif timestep is None:
            timestep = -1

        h_sequence = self._diag_output[:, timestep]
        physical_state = yb[:, timestep, :]
        h_state = self.inner_model.encode_single_state(physical_state)

        consistency_loss = F.mse_loss(h_sequence, h_state)

        self._diag_output = None

        return self.weight * consistency_loss


class TransitionSmoothnessLoss:
    """Penalizes discontinuities in predictions around the init_sz boundary.

    Args:
        init_sz: init window size (transition at this index)
        weight: loss weight
        window: timesteps around boundary to penalize
        dt: time step for derivative computation
    """

    def __init__(
        self,
        init_sz: int,
        weight: float = 1.0,
        window: int = 3,
        dt: float = 0.01,
    ):
        self.init_sz = init_sz
        self.weight = weight
        self.window = window
        self.dt = dt

    def __call__(self, pred: torch.Tensor, yb: torch.Tensor, xb: torch.Tensor) -> torch.Tensor:
        """Compute curvature penalty around the transition boundary."""
        start = max(0, self.init_sz - self.window)
        end = min(pred.shape[1], self.init_sz + self.window)
        if end - start < 3:
            return torch.tensor(0.0, device=pred.device)  # Need at least 3 points for second derivative

        y_boundary = pred[:, start:end, :]  # [batch, window_len, n_y]
        batch, wlen, ny = y_boundary.shape
        y_flat = y_boundary.permute(0, 2, 1).reshape(batch * ny, wlen)
        d2 = diff2_forward(y_flat, self.dt)
        smooth_loss = (d2**2).mean()

        return self.weight * smooth_loss
