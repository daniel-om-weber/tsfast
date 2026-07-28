"""Tests for ONNX export and inference."""

import pytest
import numpy as np

onnx = pytest.importorskip("onnx")
onnxruntime = pytest.importorskip("onnxruntime")


@pytest.fixture(scope="module")
def rnn_lrn(dls_short):
    """RNNLearner on the short-window DataLoaders, shared by the RNN export tests."""
    from tsfast.training import RNNLearner

    return RNNLearner(dls_short)


@pytest.fixture(scope="module")
def rnn_onnx(rnn_lrn, tmp_path_factory):
    """A single exported RNN, reused across the wrapper tests.

    Export cost is linear in the traced sequence length but the resulting graph is not,
    so one short export covers every test that just needs a valid model to run.
    """
    from tsfast.inference.onnx import export_onnx

    return export_onnx(rnn_lrn, tmp_path_factory.mktemp("onnx") / "model.onnx")


class TestExportOnnx:
    def test_export_rnn(self, rnn_onnx):
        assert rnn_onnx.exists()
        model = onnx.load(str(rnn_onnx))
        onnx.checker.check_model(model)

    def test_export_tcn(self, dls_simulation, tmp_path):
        from tsfast.training import TCNLearner
        from tsfast.inference.onnx import export_onnx

        lrn = TCNLearner(dls_simulation)
        path = export_onnx(lrn, tmp_path / "model.onnx")
        assert path.exists()
        model = onnx.load(str(path))
        onnx.checker.check_model(model)

    def test_export_adds_suffix(self, rnn_lrn, tmp_path):
        from tsfast.inference.onnx import export_onnx

        path = export_onnx(rnn_lrn, tmp_path / "model")
        assert path.suffix == ".onnx"
        assert path.exists()

    def test_export_ar_model_raises(self, dls_prediction, tmp_path):
        from tsfast.training import AR_RNNLearner
        from tsfast.inference.onnx import export_onnx

        lrn = AR_RNNLearner(dls_prediction)
        with pytest.raises(ValueError, match="AR_Model"):
            export_onnx(lrn, tmp_path / "model.onnx")

    def test_export_custom_seq_len(self, rnn_lrn, tmp_path):
        from tsfast.inference.onnx import export_onnx

        path = export_onnx(rnn_lrn, tmp_path / "model.onnx", seq_len=10)
        assert path.exists()


class TestOnnxInferenceWrapper:
    def test_matches_pytorch_output(self, rnn_lrn, rnn_onnx):
        from tsfast.inference.wrapper import InferenceWrapper
        from tsfast.inference.onnx import OnnxInferenceWrapper

        inp = np.random.randn(100, 1).astype(np.float32)

        pt_result = InferenceWrapper(rnn_lrn)(inp)
        onnx_result = OnnxInferenceWrapper(rnn_onnx)(inp)

        assert pt_result.shape == onnx_result.shape
        np.testing.assert_allclose(pt_result, onnx_result, atol=1e-5)

    def test_1d_input(self, rnn_onnx):
        from tsfast.inference.onnx import OnnxInferenceWrapper

        result = OnnxInferenceWrapper(rnn_onnx)(np.random.randn(100))
        assert result.shape == (100,)

    def test_2d_input(self, rnn_onnx):
        from tsfast.inference.onnx import OnnxInferenceWrapper

        result = OnnxInferenceWrapper(rnn_onnx)(np.random.randn(100, 1))
        assert result.shape == (100, 1)

    def test_3d_input(self, rnn_onnx):
        from tsfast.inference.onnx import OnnxInferenceWrapper

        result = OnnxInferenceWrapper(rnn_onnx)(np.random.randn(1, 100, 1))
        assert result.shape == (1, 100, 1)

    def test_batched_3d_input(self, rnn_lrn, rnn_onnx):
        from tsfast.inference.wrapper import InferenceWrapper
        from tsfast.inference.onnx import OnnxInferenceWrapper

        # Batched 3D input with batch_size > 1 (matches next(iter(dls.valid)) usage)
        inp = np.random.randn(4, 100, 1).astype(np.float32)
        pt_result = InferenceWrapper(rnn_lrn)(inp)
        onnx_result = OnnxInferenceWrapper(rnn_onnx)(inp)

        assert onnx_result.shape == pt_result.shape
        np.testing.assert_allclose(pt_result, onnx_result, atol=1e-5)

    def test_dynamic_seq_len(self, rnn_onnx):
        from tsfast.inference.onnx import OnnxInferenceWrapper

        wrapper = OnnxInferenceWrapper(rnn_onnx)

        # Both directions off the traced length: shorter and longer.
        r1 = wrapper(np.random.randn(10, 1))
        assert r1.shape == (10, 1)

        r2 = wrapper(np.random.randn(200, 1))
        assert r2.shape == (200, 1)

    def test_tcn_output(self, dls_simulation, tmp_path):
        from tsfast.training import TCNLearner
        from tsfast.inference.wrapper import InferenceWrapper
        from tsfast.inference.onnx import export_onnx, OnnxInferenceWrapper

        lrn = TCNLearner(dls_simulation)
        inp = np.random.randn(100, 1).astype(np.float32)

        pt_result = InferenceWrapper(lrn)(inp)

        path = export_onnx(lrn, tmp_path / "model.onnx")
        onnx_result = OnnxInferenceWrapper(path)(inp)

        assert pt_result.shape == onnx_result.shape
        np.testing.assert_allclose(pt_result, onnx_result, atol=1e-5)
