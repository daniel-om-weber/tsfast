"""Tests for tsfast.spectogram module."""
import pytest
import torch


class TestSpectrogram:
    def test_spectrogram_module_output_shape(self):
        from tsfast.spectogram import Spectrogram
        spec = Spectrogram(n_fft=64)
        waveform = torch.rand(4, 200)  # (batch, time)
        out = spec(waveform)
        assert out.shape[0] == 4
        assert out.shape[1] == 64 // 2 + 1  # freq bins

    def test_sequence2spectrogram_transform(self):
        from tsfast.spectogram import Sequence2Spectrogram, TensorSpectrogramInput
        tfm = Sequence2Spectrogram(n_fft=32)
        # Input is (batch, seq_len, channels) — Sequence2Spectrogram transposes to (batch, channels, seq_len)
        x = TensorSpectrogramInput(torch.rand(4, 200, 2))
        out = tfm(x)
        assert out.shape[0] == 4  # batch
        assert out.shape[1] == 2  # channels
        assert out.shape[2] == 32 // 2 + 1  # freq bins

    def test_spectrogram_block_from_hdf(self, wh_path):
        from tsfast.spectogram import SpectrogramBlock
        from tsfast.data.core import HDF2Sequence
        block = SpectrogramBlock.from_hdf(["u"], n_fft=64)
        assert block is not None
        assert hasattr(block, "type_tfms")
        assert hasattr(block, "batch_tfms")
