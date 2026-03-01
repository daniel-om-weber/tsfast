"""Tests for tsfast.spectogram module."""
import torch


class TestSpectrogram:
    def test_spectrogram_module_output_shape(self):
        from tsfast.spectogram import Spectrogram
        spec = Spectrogram(n_fft=64)
        waveform = torch.rand(4, 200)  # (batch, time)
        out = spec(waveform)
        assert out.shape[0] == 4
        assert out.shape[1] == 64 // 2 + 1  # freq bins

