"""Tests for signal processing utilities in tsfast.tsdata.signal."""
import numpy as np


class TestResampling:
    def test_resample_interp_shape(self):
        from tsfast.tsdata.signal import resample_interp
        x = np.random.normal(size=(100000, 9))
        assert resample_interp(x, 0.3).shape[0] == 30000


class TestDataUtilities:
    def test_running_mean_shape(self):
        from tsfast.tsdata.signal import running_mean
        x = np.random.normal(size=(100, 2))
        result = running_mean(x, 10)
        assert result.shape == (91, 2)

    def test_downsample_mean_shape(self):
        from tsfast.tsdata.signal import downsample_mean
        x = np.random.normal(size=(100, 3))
        result = downsample_mean(x, 5)
        assert result.shape == (20, 3)
