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


class TestARRNN:
    @pytest.mark.slow
    def test_ar_rnn_learner_fit(self, dls_prediction):
        from tsfast.models.rnn import AR_RNNLearner
        lrn = AR_RNNLearner(dls_prediction)
        lrn.fit(1)
        assert not math.isnan(lrn.recorder.values[-1][1])
