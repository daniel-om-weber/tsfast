"""ONNX export and inference for trained Learners."""

__all__ = ["export_onnx", "OnnxInferenceWrapper"]

import copy
from pathlib import Path

import numpy as np
import torch

from ..models.layers import AR_Model
from ..models.rnn import WeightDropout


def _check_no_ar(model):
    "Raise if model contains an AR_Model (autoregressive loops can't be exported to ONNX)."
    for module in model.modules():
        if isinstance(module, AR_Model):
            raise ValueError(
                "AR_Model (autoregressive) cannot be exported to ONNX. "
                "Export the base model instead, or use standard (non-AR) inference."
            )


def _unwrap_weight_dropout(model):
    "Replace WeightDropout wrappers with their underlying modules (weights baked in)."
    for name, module in model.named_children():
        if isinstance(module, WeightDropout):
            inner = module.module
            for layer in module.layer_names:
                raw_w = getattr(module, f"{layer}_raw")
                inner.register_parameter(layer, torch.nn.Parameter(raw_w.data))
            if hasattr(inner, "flatten_parameters"):
                inner.flatten_parameters = torch.nn.RNNBase.flatten_parameters.__get__(inner)
                inner.flatten_parameters()
            setattr(model, name, inner)
        else:
            _unwrap_weight_dropout(module)


def _fix_gru_linear_before_reset(onnx_model):
    "Work around onnxscript bug: GRU missing linear_before_reset=1. https://github.com/microsoft/onnxscript/issues/2852"
    from onnx import helper

    for node in onnx_model.graph.node:
        if node.op_type == "GRU":
            attrs = {a.name: a for a in node.attribute}
            if "linear_before_reset" not in attrs:
                node.attribute.append(helper.make_attribute("linear_before_reset", 1))
            elif attrs["linear_before_reset"].i != 1:
                attrs["linear_before_reset"].i = 1


def _get_dummy_input(learner, seq_len: int | None = None):
    "Create a dummy input tensor from the learner's dataloader shapes."
    batch = learner.dls.one_batch()
    n_features = batch[0].shape[-1]
    sl = seq_len or batch[0].shape[1]
    return torch.randn(1, sl, n_features)


def export_onnx(
    learner,  # trained Learner with .model and .dls
    path: str | Path,  # output .onnx file path
    opset_version: int = 18,  # ONNX opset version
    seq_len: int | None = None,  # override sequence length for dummy input (default: from dls)
) -> Path:
    "Export a trained Learner's model to ONNX format with normalization baked in."
    import onnx

    path = Path(path)
    if path.suffix != ".onnx":
        path = path.with_suffix(".onnx")

    model = learner.model
    _check_no_ar(model)

    model = copy.deepcopy(model).cpu().eval()
    _unwrap_weight_dropout(model)
    dummy = _get_dummy_input(learner, seq_len)

    torch.onnx.export(
        model,
        (dummy,),
        f=str(path),
        opset_version=opset_version,
        input_names=["input"],
        output_names=["output"],
        dynamic_shapes={"xb": {0: torch.export.Dim.AUTO, 1: torch.export.Dim.AUTO}},
    )

    onnx_model = onnx.load(str(path))
    _fix_gru_linear_before_reset(onnx_model)
    onnx.save(onnx_model, str(path))
    onnx.checker.check_model(onnx_model)
    return path


class OnnxInferenceWrapper:
    """Run an exported ONNX model with onnxruntime. Same API as InferenceWrapper.

    Args:
        path: path to the exported .onnx model file
        session_options: additional keyword arguments forwarded to onnxruntime.InferenceSession
    """

    def __init__(self, path: str | Path, **session_options):
        import onnxruntime as ort

        self.session = ort.InferenceSession(str(path), **session_options)
        self._input_name = self.session.get_inputs()[0].name
        self._output_name = self.session.get_outputs()[0].name

    def _prepare(self, np_array: np.ndarray, name: str) -> np.ndarray:
        "Reshape to [batch, seq_len, features] float32."
        if not isinstance(np_array, np.ndarray):
            raise TypeError(f"{name} must be a NumPy array.")
        if np_array.ndim == 1:
            np_array = np_array[None, :, None]
        elif np_array.ndim == 2:
            np_array = np_array[None, :, :]
        elif np_array.ndim != 3:
            raise ValueError(f"{name} must be 1D, 2D, or 3D. Got {np_array.ndim}D with shape: {np_array.shape}")
        return np_array.astype(np.float32)

    def inference(
        self,
        np_input: np.ndarray,  # input time series
        np_output_init: np.ndarray | None = None,  # initial output (for PredictionCallback models)
    ) -> np.ndarray:
        "Run inference on numpy input, returns numpy output. Output ndim mirrors input ndim."
        input_ndim = np_input.ndim
        u = self._prepare(np_input, "np_input")
        if np_output_init is not None:
            y_init = self._prepare(np_output_init, "np_output_init")
            if u.shape[0] != y_init.shape[0]:
                raise ValueError(
                    f"Batch size mismatch: np_input has {u.shape[0]}, np_output_init has {y_init.shape[0]}."
                )
            seq_len = u.shape[1]
            if y_init.shape[1] != seq_len:
                y_init = (
                    y_init[:, :seq_len, :]
                    if y_init.shape[1] > seq_len
                    else np.pad(y_init, ((0, 0), (0, seq_len - y_init.shape[1]), (0, 0)))
                )
            u = np.concatenate((u, y_init), axis=-1)

        result = self.session.run([self._output_name], {self._input_name: u})[0]
        match input_ndim:
            case 1:
                if result.shape[-1] != 1:
                    raise ValueError(
                        f"Cannot return 1D output: model produces {result.shape[-1]} features. "
                        f"Pass 2D input (seq_len, features) instead."
                    )
                return result[0, :, 0]
            case 2:
                return result[0]
            case _:
                return result

    def __call__(self, np_input: np.ndarray, np_output_init: np.ndarray | None = None) -> np.ndarray:
        return self.inference(np_input, np_output_init)
