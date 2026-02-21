
__all__ = ['InferenceWrapper']

from ..data.loader import reset_model_state
import warnings
from ..prediction.core import PredictionCallback
from fastai.learner import Learner
import numpy as np
import torch

class InferenceWrapper:
    def __init__(self,
                 learner: Learner, # The trained fastai Learner object.
                 device: str | torch.device = 'cpu' # Device for inference ('cpu', 'cuda').
                ):
        if not isinstance(learner, Learner) or not hasattr(learner, 'model') or not hasattr(learner, 'dls'): raise TypeError("Input 'learner' must be a valid fastai Learner with model and dls.")
        self.device = torch.device(device)
        self.model = learner.model.to(self.device).eval()
        self._pred_cb: PredictionCallback | None = next((cb for cb in learner.cbs if isinstance(cb, PredictionCallback)), None)

    def _prepare_tensor(self, np_array: np.ndarray, name: str) -> torch.Tensor:
        'Converts numpy array to a 3D tensor [1, seq_len, features] on the correct device.'
        if not isinstance(np_array, np.ndarray): raise TypeError(f"{name} must be a NumPy array.")
        if np_array.ndim == 1: np_array = np_array[None, :, None]
        elif np_array.ndim == 2: np_array = np_array[None, :, :]
        elif np_array.ndim != 3 or np_array.shape[0] != 1: raise ValueError(f"{name} must be 1D, 2D, or 3D with batch_size=1. Got shape: {np_array.shape}")
        return torch.from_numpy(np_array).float().to(self.device)

    def _adjust_seq_len(self, tensor: torch.Tensor, target_len: int, name: str) -> torch.Tensor:
        'Adjusts sequence length (dim 1) of a [1, seq_len, features] tensor.'
        current_len = tensor.shape[1]
        if current_len == target_len: return tensor
        if current_len < target_len: return torch.nn.functional.pad(tensor, (0, 0, 0, target_len - current_len))
        warnings.warn(f"Truncating {name} seq len from {current_len} to {target_len}.", UserWarning)
        return tensor[:, :target_len, :]

    @torch.no_grad()
    def inference(self,
                  np_input: np.ndarray, # Input time series (u). Shape [seq_len], [seq_len, u_features], or [1, seq_len, u_features].
                  np_output_init: np.ndarray | None = None # Initial output series (y_init). Required if trained with PredictionCallback. Defaults to None.
                 ) -> np.ndarray:
        u_tensor = self._prepare_tensor(np_input, "np_input")
        input_seq_len = u_tensor.shape[1]
        y_init_tensor = self._prepare_tensor(np_output_init, "np_output_init") if np_output_init is not None else None

        if self._pred_cb:
            if y_init_tensor is None: raise ValueError("Model trained with PredictionCallback requires 'np_output_init'.")
            if input_seq_len - self._pred_cb.t_offset <= 0: raise ValueError(f"Input seq len ({input_seq_len}) too short for offset ({self._pred_cb.t_offset}).")
            if self._pred_cb.t_offset > 0:
                u_tensor = u_tensor[:, self._pred_cb.t_offset:, :]
                y_init_tensor = y_init_tensor[:, :-self._pred_cb.t_offset, :]
            y_init_tensor = self._adjust_seq_len(y_init_tensor, input_seq_len, "y_init")
            final_input = torch.cat((u_tensor, y_init_tensor), dim=-1)
        elif y_init_tensor is not None:
            y_init_tensor = self._adjust_seq_len(y_init_tensor, input_seq_len, "y_init")
            final_input = torch.cat((u_tensor, y_init_tensor), dim=-1)
        else:
            final_input = u_tensor

        reset_model_state(self.model)
        model_output = self.model(final_input)
        output_tensor = model_output[0] if isinstance(model_output, tuple) else model_output
        if not isinstance(output_tensor, torch.Tensor): raise RuntimeError(f"Model output is not a tensor. Type: {type(output_tensor)}")
        return output_tensor.squeeze(0).cpu().numpy()

    def __call__(self, np_input: np.ndarray, np_output_init: np.ndarray | None = None) -> np.ndarray:
        return self.inference(np_input, np_output_init)
