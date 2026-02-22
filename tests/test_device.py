"""Accelerator device-mismatch regression tests.

Runs on any available non-CPU accelerator (CUDA, MPS, XPU, …).
Skipped when only CPU is available.
"""
import math
import pytest
import torch
from torch import nn


def _get_accelerator() -> torch.device | None:
    """Return the first available non-CPU accelerator, or None."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    return None


_accelerator = _get_accelerator()

requires_accelerator = pytest.mark.skipif(
    _accelerator is None, reason="No non-CPU accelerator available"
)


@pytest.fixture
def device():
    """The first available non-CPU device."""
    return _accelerator


@requires_accelerator
class TestHookCallbackDevices:
    """HookCallback subclasses must keep hooked tensors on the model device."""

    @pytest.mark.slow
    def test_time_series_regularizer(self, dls_simulation, device):
        from tsfast.models.rnn import SimpleRNN
        from tsfast.learner.callbacks import TimeSeriesRegularizer
        from fastai.basics import Learner

        model = SimpleRNN(1, 1, hidden_size=20).to(device)
        lrn = Learner(
            dls_simulation, model, loss_func=nn.MSELoss(),
            cbs=TimeSeriesRegularizer(alpha=0.1, beta=0.1, modules=[model.rnn]),
        )
        lrn.fit(1)
        final_valid_loss = lrn.recorder.values[-1][1]
        assert final_valid_loss < float("inf")
        assert not math.isnan(final_valid_loss)

    @pytest.mark.slow
    def test_fransys_callback(self, dls_prediction, device):
        from tsfast.prediction.fransys import FranSysLearner, FranSysCallback
        from tsfast.models.layers import unwrap_model

        lrn = FranSysLearner(dls_prediction, init_sz=50, hidden_size=20, rnn_layer=1)
        lrn.model.to(device)
        model = unwrap_model(lrn.model)
        lrn.add_cb(FranSysCallback(
            modules=[model.rnn_diagnosis, model.rnn_prognosis],
            p_state_sync=1.0, sync_type="mse", model=model,
        ))
        lrn.fit(1, 3e-3)
        assert not math.isnan(lrn.recorder.values[-1][1])

    @pytest.mark.slow
    def test_quaternion_regularizer(self, dls_simulation, device):
        from tsfast.models.rnn import SimpleRNN
        from tsfast.quaternions import QuaternionRegularizer
        from fastai.basics import Learner

        # SimpleRNN with output_size=4 to mimic quaternion output
        model = SimpleRNN(1, 4, hidden_size=20).to(device)
        lrn = Learner(
            dls_simulation, model, loss_func=nn.MSELoss(),
            cbs=QuaternionRegularizer(reg_unit=0.1, modules=[model.rnn]),
        )
        lrn.fit(1)
        final_valid_loss = lrn.recorder.values[-1][1]
        assert final_valid_loss < float("inf")
        assert not math.isnan(final_valid_loss)


@requires_accelerator
class TestTransformDevices:
    """Data transforms must sync internal tensors to the input device."""

    def test_seq_noise_injection_grouped(self, device):
        from tsfast.data.transforms import SeqNoiseInjection_Grouped
        from tsfast.data.core import TensorSequencesInput

        tfm = SeqNoiseInjection_Grouped(std_std=[0.1, 0.2], std_idx=[0, 1, 0])
        x = TensorSequencesInput(torch.randn(2, 50, 3)).to(device)
        # Should not raise a device mismatch error
        out = tfm(x)
        assert out.device == x.device


@requires_accelerator
class TestQuaternionMathDevices:
    """Quaternion math helpers must work on non-CPU tensors."""

    def test_inclination_angle_abs(self, device):
        from tsfast.quaternions import inclinationAngleAbs

        q = torch.tensor([[1.0, 0, 0, 0], [0.707, 0.707, 0, 0]], device=device)
        result = inclinationAngleAbs(q)
        assert result.device.type == device.type
        assert result.shape == (2,)


@requires_accelerator
class TestQuaternionPlotDevices:
    """Quaternion plot functions must handle non-CPU tensors (via .cpu())."""

    def _make_axes(self, n=3):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        _, axs = plt.subplots(n, 1)
        return axs

    def test_plot_scalar_inclination(self, device):
        from tsfast.quaternions import plot_scalar_inclination

        axs = self._make_axes(3)
        targ = torch.randn(50, device=device)
        out = torch.randn(50, device=device)
        inp = torch.randn(50).numpy()  # in_sig is typically numpy
        plot_scalar_inclination(axs, inp, targ, out)

    def test_plot_quaternion_inclination(self, device):
        from tsfast.quaternions import plot_quaternion_inclination

        axs = self._make_axes(3)
        targ = torch.randn(50, 4, device=device)
        targ = targ / targ.norm(dim=-1, keepdim=True)
        out = torch.randn(50, 4, device=device)
        out = out / out.norm(dim=-1, keepdim=True)
        inp = torch.randn(50).numpy()
        plot_quaternion_inclination(axs, inp, targ, out)

    def test_plot_quaternion_rel_angle(self, device):
        from tsfast.quaternions import plot_quaternion_rel_angle

        axs = self._make_axes(3)
        targ = torch.randn(50, 4, device=device)
        targ = targ / targ.norm(dim=-1, keepdim=True)
        out = torch.randn(50, 4, device=device)
        out = out / out.norm(dim=-1, keepdim=True)
        inp = torch.randn(50).numpy()
        plot_quaternion_rel_angle(axs, inp, targ, out)
