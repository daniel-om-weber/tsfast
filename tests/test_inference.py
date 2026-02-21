"""Tests for tsfast.inference module."""
import pytest
import numpy as np


class TestInferenceWrapper:
    def test_simulation_2d_input(self, dls_simulation):
        from tsfast.models.rnn import RNNLearner
        from tsfast.inference.core import InferenceWrapper
        lrn = RNNLearner(dls_simulation)
        model = InferenceWrapper(lrn)
        result = model(np.random.randn(100, 1))
        assert result.shape == (100, 1)

    def test_simulation_1d_input(self, dls_simulation):
        from tsfast.models.rnn import RNNLearner
        from tsfast.inference.core import InferenceWrapper
        lrn = RNNLearner(dls_simulation)
        model = InferenceWrapper(lrn)
        result = model(np.random.randn(100))
        assert result.shape == (100, 1)

    def test_simulation_3d_input(self, dls_simulation):
        from tsfast.models.rnn import RNNLearner
        from tsfast.inference.core import InferenceWrapper
        lrn = RNNLearner(dls_simulation)
        model = InferenceWrapper(lrn)
        result = model(np.random.randn(1, 100, 1))
        assert result.shape == (100, 1)

    @pytest.mark.slow
    def test_prediction_with_output_init(self, dls_simulation):
        from tsfast.prediction.fransys import FranSysLearner
        from tsfast.inference.core import InferenceWrapper
        lrn = FranSysLearner(dls_simulation, 10, attach_output=True)
        model = InferenceWrapper(lrn)
        result = model(np.random.randn(100, 1), np.random.randn(100, 1))
        assert result.shape == (100, 1)
        assert np.all(np.isfinite(result))
