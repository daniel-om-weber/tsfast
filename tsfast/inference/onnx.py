"""ONNX export and inference for trained Learners."""

__all__ = ["export_onnx", "OnnxInferenceWrapper"]

from pathlib import Path

import numpy as np
import torch
from fastai.learner import Learner

from ..models.layers import AR_Model


def _check_no_ar(model):
    "Raise if model contains an AR_Model (autoregressive loops can't be exported to ONNX)."
    for module in model.modules():
        if isinstance(module, AR_Model):
            raise ValueError(
                "AR_Model (autoregressive) cannot be exported to ONNX. "
                "Export the base model instead, or use standard (non-AR) inference."
            )


def _get_dummy_input(learner, seq_len: int | None = None):
    "Create a dummy input tensor from the learner's dataloader shapes."
    batch = learner.dls.one_batch()
    n_features = batch[0].shape[-1]
    sl = seq_len or batch[0].shape[1]
    return torch.randn(1, sl, n_features)


def export_onnx(
    learner: Learner,  # trained fastai Learner
    path: str | Path,  # output .onnx file path
    opset_version: int = 17,  # ONNX opset version
    seq_len: int | None = None,  # override sequence length for dummy input (default: from dls)
) -> Path:
    "Export a trained Learner's model to ONNX format with normalization baked in."
    import onnx

    path = Path(path)
    if path.suffix != ".onnx":
        path = path.with_suffix(".onnx")

    model = learner.model
    _check_no_ar(model)

    model = model.cpu().eval()
    dummy = _get_dummy_input(learner, seq_len)

    torch.onnx.export(
        model,
        (dummy,),
        f=str(path),
        opset_version=opset_version,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch", 1: "seq_len"},
            "output": {0: "batch", 1: "seq_len"},
        },
        dynamo=False,
    )

    onnx_model = onnx.load(str(path))
    onnx.checker.check_model(onnx_model)
    return path


class OnnxInferenceWrapper:
    "Run an exported ONNX model with onnxruntime. Same API as InferenceWrapper."

    def __init__(self, path: str | Path, **session_options):
        import onnxruntime as ort

        self.session = ort.InferenceSession(str(path), **session_options)
        self._input_name = self.session.get_inputs()[0].name
        self._output_name = self.session.get_outputs()[0].name

    def _prepare(self, np_array: np.ndarray, name: str) -> np.ndarray:
        "Reshape to [1, seq_len, features] float32."
        if not isinstance(np_array, np.ndarray):
            raise TypeError(f"{name} must be a NumPy array.")
        if np_array.ndim == 1:
            np_array = np_array[None, :, None]
        elif np_array.ndim == 2:
            np_array = np_array[None, :, :]
        elif np_array.ndim != 3 or np_array.shape[0] != 1:
            raise ValueError(f"{name} must be 1D, 2D, or 3D with batch_size=1. Got shape: {np_array.shape}")
        return np_array.astype(np.float32)

    def inference(
        self,
        np_input: np.ndarray,  # input time series
        np_output_init: np.ndarray | None = None,  # initial output (for PredictionCallback models)
    ) -> np.ndarray:
        "Run inference on numpy input, returns numpy output."
        u = self._prepare(np_input, "np_input")
        if np_output_init is not None:
            y_init = self._prepare(np_output_init, "np_output_init")
            seq_len = u.shape[1]
            if y_init.shape[1] != seq_len:
                y_init = (
                    y_init[:, :seq_len, :]
                    if y_init.shape[1] > seq_len
                    else np.pad(y_init, ((0, 0), (0, seq_len - y_init.shape[1]), (0, 0)))
                )
            u = np.concatenate((u, y_init), axis=-1)

        result = self.session.run([self._output_name], {self._input_name: u})[0]
        return result.squeeze(0)

    def __call__(self, np_input: np.ndarray, np_output_init: np.ndarray | None = None) -> np.ndarray:
        return self.inference(np_input, np_output_init)
