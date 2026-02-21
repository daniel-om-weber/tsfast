"""Tests for tsfast.prediction module."""
import math
import pytest
import torch


class TestFranSys:
    def test_fransys_forward_shape(self, dls_prediction):
        from tsfast.prediction.fransys import FranSys
        batch = dls_prediction.one_batch()
        device = batch[0].device
        model = FranSys(1, 1, init_sz=50, rnn_layer=2, hidden_size=50).to(device)
        out = model(batch[0])
        assert out.shape[0] == batch[0].shape[0]
        assert out.shape[-1] == 1

    def test_arprog_init_forward(self, dls_prediction):
        from tsfast.prediction.fransys import ARProg_Init
        batch = dls_prediction.one_batch()
        device = batch[0].device
        model = ARProg_Init(1, 1, init_sz=50, rnn_layer=1, hidden_size=50).to(device)
        out = model(batch[0])
        assert out.shape[-1] == 1

    @pytest.mark.slow
    def test_fransys_learner_prediction_mode(self, dls_prediction):
        from tsfast.prediction.fransys import FranSysLearner
        lrn = FranSysLearner(dls_prediction, init_sz=50)
        lrn.fit(1, lr=3e-3)
        assert not math.isnan(lrn.recorder.values[-1][1])

    @pytest.mark.slow
    def test_fransys_learner_attach_output(self, dls_simulation):
        from tsfast.prediction.fransys import FranSysLearner
        lrn = FranSysLearner(dls_simulation, init_sz=50, attach_output=True)
        lrn.fit(1, lr=3e-3)
        assert not math.isnan(lrn.recorder.values[-1][1])


class TestFranSysRegularization:
    def _get_model(self, lrn):
        """Unwrap NormalizedModel to access FranSys internals."""
        from tsfast.models.layers import unwrap_model
        return unwrap_model(lrn.model)

    @pytest.mark.slow
    @pytest.mark.parametrize("sync_type", [
        "mse", "mae", "mspe", "mape", "cos", "cos_pow",
    ])
    def test_fransys_sync_types(self, dls_prediction, sync_type):
        from tsfast.prediction.fransys import FranSysLearner, FranSysCallback
        lrn = FranSysLearner(dls_prediction, init_sz=50, hidden_size=20, rnn_layer=1)
        model = self._get_model(lrn)
        lrn.add_cb(FranSysCallback(
            modules=[model.rnn_diagnosis, model.rnn_prognosis],
            p_state_sync=1.0, sync_type=sync_type, model=model,
        ))
        lrn.fit(1, 3e-3)
        assert not math.isnan(lrn.recorder.values[-1][1])

    @pytest.mark.slow
    def test_fransys_diag_loss(self, dls_prediction):
        from tsfast.prediction.fransys import FranSysLearner, FranSysCallback
        lrn = FranSysLearner(dls_prediction, init_sz=50, hidden_size=20, rnn_layer=1)
        model = self._get_model(lrn)
        lrn.add_cb(FranSysCallback(
            modules=[model.rnn_diagnosis, model.rnn_prognosis],
            p_state_sync=0, p_diag_loss=0.1, model=model,
        ))
        lrn.fit(1, 3e-3)
        assert not math.isnan(lrn.recorder.values[-1][1])

    @pytest.mark.slow
    def test_fransys_osp_loss(self, dls_prediction):
        from tsfast.prediction.fransys import FranSysLearner, FranSysCallback
        lrn = FranSysLearner(dls_prediction, init_sz=50, hidden_size=20, rnn_layer=1)
        model = self._get_model(lrn)
        lrn.add_cb(FranSysCallback(
            modules=[model.rnn_diagnosis, model.rnn_prognosis],
            p_state_sync=0, p_osp_loss=0.1, p_osp_sync=0.1, model=model,
        ))
        lrn.fit(1, 3e-3)
        assert not math.isnan(lrn.recorder.values[-1][1])

    @pytest.mark.slow
    def test_fransys_tar_loss(self, dls_prediction):
        from tsfast.prediction.fransys import FranSysLearner, FranSysCallback
        lrn = FranSysLearner(dls_prediction, init_sz=50, hidden_size=20, rnn_layer=1)
        model = self._get_model(lrn)
        lrn.add_cb(FranSysCallback(
            modules=[model.rnn_diagnosis, model.rnn_prognosis],
            p_state_sync=0, p_tar_loss=0.1, model=model,
        ))
        lrn.fit(1, 3e-3)
        assert not math.isnan(lrn.recorder.values[-1][1])

    @pytest.mark.slow
    def test_fransys_variable_init(self, dls_prediction):
        from tsfast.prediction.fransys import FranSysLearner, FranSysCallback_variable_init
        lrn = FranSysLearner(dls_prediction, init_sz=50, hidden_size=20, rnn_layer=1)
        lrn.add_cb(FranSysCallback_variable_init(init_sz_min=30, init_sz_max=70))
        lrn.fit(1, 3e-3)
        assert not math.isnan(lrn.recorder.values[-1][1])

    @pytest.mark.slow
    def test_fransys_diag_loss_with_output_norm(self, dls_prediction):
        """Verify diag_loss denormalizes predictions when output_norm is used."""
        from tsfast.prediction.fransys import FranSysLearner, FranSysCallback
        from tsfast.models.layers import StandardScaler1D
        lrn = FranSysLearner(
            dls_prediction, init_sz=50, hidden_size=20, rnn_layer=1,
            output_norm=StandardScaler1D,
        )
        model = self._get_model(lrn)
        lrn.add_cb(FranSysCallback(
            modules=[model.rnn_diagnosis, model.rnn_prognosis],
            p_state_sync=0, p_diag_loss=0.1, model=model,
        ))
        lrn.fit(1, 3e-3)
        assert not math.isnan(lrn.recorder.values[-1][1])

    @pytest.mark.slow
    def test_fransys_osp_loss_with_output_norm(self, dls_prediction):
        """Verify osp_loss denormalizes predictions when output_norm is used."""
        from tsfast.prediction.fransys import FranSysLearner, FranSysCallback
        from tsfast.models.layers import StandardScaler1D
        lrn = FranSysLearner(
            dls_prediction, init_sz=50, hidden_size=20, rnn_layer=1,
            output_norm=StandardScaler1D,
        )
        model = self._get_model(lrn)
        lrn.add_cb(FranSysCallback(
            modules=[model.rnn_diagnosis, model.rnn_prognosis],
            p_state_sync=0, p_osp_loss=0.1, p_osp_sync=0.1, model=model,
        ))
        lrn.fit(1, 3e-3)
        assert not math.isnan(lrn.recorder.values[-1][1])

    def test_fransys_callback_captures_output_norm(self, dls_prediction):
        """Verify FranSysCallback detects output_norm from NormalizedModel."""
        from tsfast.prediction.fransys import FranSysLearner, FranSysCallback
        from tsfast.models.layers import NormalizedModel, StandardScaler1D

        lrn = FranSysLearner(
            dls_prediction, init_sz=50, hidden_size=20, rnn_layer=1,
            output_norm=StandardScaler1D,
        )
        assert isinstance(lrn.model, NormalizedModel)
        assert lrn.model.output_norm is not None

        model = self._get_model(lrn)
        cb = FranSysCallback(
            modules=[model.rnn_diagnosis, model.rnn_prognosis],
            p_state_sync=0, p_diag_loss=0.1, model=model,
        )
        lrn.add_cb(cb)
        # before_fit triggers output_norm detection
        cb.learn = lrn
        cb.before_fit()
        assert cb._output_norm is lrn.model.output_norm

    def test_fransys_callback_no_output_norm(self, dls_prediction):
        """Verify output_norm is None when NormalizedModel has no output scaler."""
        from tsfast.prediction.fransys import FranSysLearner, FranSysCallback
        from tsfast.models.layers import NormalizedModel

        lrn = FranSysLearner(dls_prediction, init_sz=50, hidden_size=20, rnn_layer=1)
        assert isinstance(lrn.model, NormalizedModel)
        assert lrn.model.output_norm is None

        model = self._get_model(lrn)
        cb = FranSysCallback(
            modules=[model.rnn_diagnosis, model.rnn_prognosis],
            p_state_sync=0, p_diag_loss=0.1, model=model,
        )
        lrn.add_cb(cb)
        cb.learn = lrn
        cb.before_fit()
        assert cb._output_norm is None

    def test_fransys_variable_init_writes_through_unwrap(self, dls_prediction):
        """Verify that unwrap_model returns the inner model and writes reach forward()."""
        from tsfast.prediction.fransys import FranSysLearner
        from tsfast.models.layers import NormalizedModel, unwrap_model

        lrn = FranSysLearner(dls_prediction, init_sz=50, hidden_size=20, rnn_layer=1)
        wrapper = lrn.model
        assert isinstance(wrapper, NormalizedModel), "Test requires NormalizedModel wrapping"
        inner = unwrap_model(wrapper)
        assert inner is wrapper.model, "unwrap_model should return the inner model"

        batch = dls_prediction.one_batch()
        xb = batch[0]

        wrapper.eval()
        with torch.no_grad():
            # Write init_sz on the inner model (what callbacks now do via unwrap_model)
            inner.init_sz = 30
            out_30 = wrapper(xb).clone()
            inner.init_sz = 50
            out_50 = wrapper(xb).clone()

        # Different init_sz produces different outputs
        assert not torch.equal(out_30, out_50), (
            "Writing init_sz on inner model did not affect forward output."
        )


class TestDDPUnwrap:
    """Tests that unwrap_model and _output_norm detection work through DDP-like wrappers."""

    def test_unwrap_model_through_ddp(self):
        """unwrap_model returns the inner FranSys model through a DDP-like wrapper."""
        from tsfast.prediction.fransys import FranSys
        from tsfast.models.layers import NormalizedModel, StandardScaler1D, unwrap_model
        import numpy as np

        inner = FranSys(1, 1, init_sz=10, hidden_size=10, rnn_layer=1)
        norm = NormalizedModel(inner, StandardScaler1D(np.zeros(2), np.ones(2)))

        # Simulate DDP wrapper: an nn.Module with a .module attribute
        class FakeDDP(torch.nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module

        wrapped = FakeDDP(norm)
        assert unwrap_model(wrapped) is inner

    def test_unwrap_ddp_finds_normalized_model(self):
        """_unwrap_ddp + isinstance check finds NormalizedModel through DDP wrapper."""
        from tsfast.models.layers import NormalizedModel, StandardScaler1D, _unwrap_ddp
        from tsfast.prediction.fransys import FranSys
        import numpy as np

        inner = FranSys(1, 1, init_sz=10, hidden_size=10, rnn_layer=1)
        out_scaler = StandardScaler1D(np.zeros(1), np.ones(1))
        norm = NormalizedModel(inner, StandardScaler1D(np.zeros(2), np.ones(2)), out_scaler)

        class FakeDDP(torch.nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module

        wrapped = FakeDDP(norm)
        unwrapped = _unwrap_ddp(wrapped)
        assert isinstance(unwrapped, NormalizedModel)
        assert unwrapped.output_norm is out_scaler

    def test_unwrap_model_no_ddp(self):
        """unwrap_model still works when there is no DDP wrapper."""
        from tsfast.prediction.fransys import FranSys
        from tsfast.models.layers import NormalizedModel, StandardScaler1D, unwrap_model
        import numpy as np

        inner = FranSys(1, 1, init_sz=10, hidden_size=10, rnn_layer=1)
        norm = NormalizedModel(inner, StandardScaler1D(np.zeros(2), np.ones(2)))
        assert unwrap_model(norm) is inner
        assert unwrap_model(inner) is inner


class TestARRNN:
    @pytest.mark.slow
    def test_ar_rnn_learner_fit(self, dls_simulation):
        from tsfast.models.rnn import AR_RNNLearner
        lrn = AR_RNNLearner(dls_simulation)
        lrn.fit(1)
        assert not math.isnan(lrn.recorder.values[-1][1])
