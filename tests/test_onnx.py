"""Tests for ONNX export and inference."""
import pytest
import numpy as np

onnx = pytest.importorskip("onnx")
onnxruntime = pytest.importorskip("onnxruntime")


class TestExportOnnx:
    def test_export_rnn(self, dls_simulation, tmp_path):
        from tsfast.models.rnn import RNNLearner
        from tsfast.inference.onnx import export_onnx

        lrn = RNNLearner(dls_simulation)
        path = export_onnx(lrn, tmp_path / "model.onnx")
        assert path.exists()
        model = onnx.load(str(path))
        onnx.checker.check_model(model)

    def test_export_tcn(self, dls_simulation, tmp_path):
        from tsfast.models.cnn import TCNLearner
        from tsfast.inference.onnx import export_onnx

        lrn = TCNLearner(dls_simulation)
        path = export_onnx(lrn, tmp_path / "model.onnx")
        assert path.exists()
        model = onnx.load(str(path))
        onnx.checker.check_model(model)

    def test_export_adds_suffix(self, dls_simulation, tmp_path):
        from tsfast.models.rnn import RNNLearner
        from tsfast.inference.onnx import export_onnx

        lrn = RNNLearner(dls_simulation)
        path = export_onnx(lrn, tmp_path / "model")
        assert path.suffix == ".onnx"
        assert path.exists()

    def test_export_ar_model_raises(self, dls_prediction, tmp_path):
        from tsfast.models.rnn import AR_RNNLearner
        from tsfast.inference.onnx import export_onnx

        lrn = AR_RNNLearner(dls_prediction)
        with pytest.raises(ValueError, match="AR_Model"):
            export_onnx(lrn, tmp_path / "model.onnx")

    def test_export_custom_seq_len(self, dls_simulation, tmp_path):
        from tsfast.models.rnn import RNNLearner
        from tsfast.inference.onnx import export_onnx

        lrn = RNNLearner(dls_simulation)
        path = export_onnx(lrn, tmp_path / "model.onnx", seq_len=50)
        assert path.exists()


class TestOnnxInferenceWrapper:
    def test_matches_pytorch_output(self, dls_simulation, tmp_path):
        from tsfast.models.rnn import RNNLearner
        from tsfast.inference.core import InferenceWrapper
        from tsfast.inference.onnx import export_onnx, OnnxInferenceWrapper

        lrn = RNNLearner(dls_simulation)
        inp = np.random.randn(100, 1).astype(np.float32)

        pt_result = InferenceWrapper(lrn)(inp)

        path = export_onnx(lrn, tmp_path / "model.onnx")
        onnx_result = OnnxInferenceWrapper(path)(inp)

        assert pt_result.shape == onnx_result.shape
        np.testing.assert_allclose(pt_result, onnx_result, atol=1e-5)

    def test_1d_input(self, dls_simulation, tmp_path):
        from tsfast.models.rnn import RNNLearner
        from tsfast.inference.onnx import export_onnx, OnnxInferenceWrapper

        lrn = RNNLearner(dls_simulation)
        path = export_onnx(lrn, tmp_path / "model.onnx")
        result = OnnxInferenceWrapper(path)(np.random.randn(100))
        assert result.shape == (100, 1)

    def test_2d_input(self, dls_simulation, tmp_path):
        from tsfast.models.rnn import RNNLearner
        from tsfast.inference.onnx import export_onnx, OnnxInferenceWrapper

        lrn = RNNLearner(dls_simulation)
        path = export_onnx(lrn, tmp_path / "model.onnx")
        result = OnnxInferenceWrapper(path)(np.random.randn(100, 1))
        assert result.shape == (100, 1)

    def test_3d_input(self, dls_simulation, tmp_path):
        from tsfast.models.rnn import RNNLearner
        from tsfast.inference.onnx import export_onnx, OnnxInferenceWrapper

        lrn = RNNLearner(dls_simulation)
        path = export_onnx(lrn, tmp_path / "model.onnx")
        result = OnnxInferenceWrapper(path)(np.random.randn(1, 100, 1))
        assert result.shape == (100, 1)

    def test_dynamic_seq_len(self, dls_simulation, tmp_path):
        from tsfast.models.rnn import RNNLearner
        from tsfast.inference.onnx import export_onnx, OnnxInferenceWrapper

        lrn = RNNLearner(dls_simulation)
        path = export_onnx(lrn, tmp_path / "model.onnx")
        wrapper = OnnxInferenceWrapper(path)

        r1 = wrapper(np.random.randn(50, 1))
        assert r1.shape == (50, 1)

        r2 = wrapper(np.random.randn(200, 1))
        assert r2.shape == (200, 1)

    def test_tcn_output(self, dls_simulation, tmp_path):
        from tsfast.models.cnn import TCNLearner
        from tsfast.inference.core import InferenceWrapper
        from tsfast.inference.onnx import export_onnx, OnnxInferenceWrapper

        lrn = TCNLearner(dls_simulation)
        inp = np.random.randn(100, 1).astype(np.float32)

        pt_result = InferenceWrapper(lrn)(inp)

        path = export_onnx(lrn, tmp_path / "model.onnx")
        onnx_result = OnnxInferenceWrapper(path)(inp)

        assert pt_result.shape == onnx_result.shape
        np.testing.assert_allclose(pt_result, onnx_result, atol=1e-5)
