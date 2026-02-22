"""Spectrogram transforms and data blocks for frequency-domain analysis."""

__all__ = [
    "TensorSpectrogram",
    "TensorSpectrogramInput",
    "TensorSpectrogramOutput",
    "Sequence2Spectrogram",
    "SpectrogramBlock",
]

from collections.abc import Callable

import matplotlib.pyplot as plt
import torch
from torch import Tensor

from fastcore.meta import delegates
from fastcore.basics import ifnone
from fastai.torch_basics import TensorBase, Transform
from fastai.data.block import TransformBlock

from .data.core import HDF2Sequence
from .data.block import pad_sequence


class TensorSpectrogram(TensorBase):
    """Base tensor type for spectrogram data with plotting support."""

    def show(self, ctx=None, ax=None, title: str = "", **kwargs):
        ax = ifnone(ax, ctx)
        if ax is None:
            _, ax = plt.subplots()
        ax.axis(False)
        n_channels = self.shape[0]
        for i, channel in enumerate(self):
            ia = ax.inset_axes((i / n_channels, 0.2, 1 / n_channels, 0.7))
            #             ia = ax.inset_axes((i / n_channels, 0, 1 / n_channels, 1))

            ia.imshow(channel.cpu().numpy(), aspect="auto", origin="lower")
            if i > 0:
                ia.set_yticks([])
            ia.set_title(f"Channel {i}")
        ax.set_title(title)
        return ax


class TensorSpectrogramInput(TensorSpectrogram):
    pass


class TensorSpectrogramOutput(TensorSpectrogram):
    pass


def complex_norm(complex_tensor: Tensor, power: float = 1.0) -> Tensor:
    """Compute the norm of a complex tensor raised to a power."""
    if power == 1.0:
        return torch.norm(complex_tensor, 2, -1)
    return torch.norm(complex_tensor, 2, -1).pow(power)


def spectrogram(
    waveform: Tensor,
    pad: int,
    window: Tensor,
    n_fft: int,
    hop_length: int,
    win_length: int,
    power: float | None,
    normalized: bool,
) -> Tensor:
    """Compute a spectrogram from an audio/signal waveform.

    Args:
        waveform: input signal tensor of shape (..., time).
        pad: two-sided zero-padding to apply.
        window: window tensor for STFT.
        n_fft: FFT size, creates n_fft // 2 + 1 frequency bins.
        hop_length: hop between STFT windows.
        win_length: window size for STFT.
        power: exponent for magnitude spectrogram, or None for complex.
        normalized: whether to normalize by window magnitude after STFT.

    Returns:
        Spectrogram tensor of shape (..., freq, time).
    """
    if pad > 0:
        waveform = torch.nn.functional.pad(waveform, (pad, pad), "constant")

    # pack batch
    shape = waveform.size()
    waveform = waveform.view(-1, shape[-1])

    # default values are consistent with librosa.core.spectrum._spectrogram
    spec_f = torch.view_as_real(
        torch.stft(waveform, n_fft, hop_length, win_length, window, True, "reflect", False, True, return_complex=True)
    )

    # unpack batch
    spec_f = spec_f.view(shape[:-1] + spec_f.shape[-3:])

    if normalized:
        spec_f /= window.pow(2.0).sum().sqrt()
    if power is not None:
        spec_f = complex_norm(spec_f, power=power)

    return spec_f


class Spectrogram(torch.nn.Module):
    """Create a spectrogram from an audio signal.

    Args:
        n_fft: size of FFT, creates ``n_fft // 2 + 1`` bins.
        win_length: window size, defaults to ``n_fft``.
        hop_length: hop between STFT windows, defaults to ``win_length // 2``.
        pad: two-sided padding of signal.
        window_fn: callable that creates a window tensor for each frame.
        power: exponent for the magnitude spectrogram (e.g. 1 for energy,
            2 for power), or None for complex spectrum.
        normalized: whether to normalize by magnitude after STFT.
        wkwargs: additional keyword arguments for the window function.
    """

    __constants__ = ["n_fft", "win_length", "hop_length", "pad", "power", "normalized"]

    def __init__(
        self,
        n_fft: int = 400,
        win_length: int | None = None,
        hop_length: int | None = None,
        pad: int = 0,
        window_fn: Callable[..., Tensor] = torch.hann_window,
        power: float | None = 2.0,
        normalized: bool = False,
        wkwargs: dict | None = None,
    ) -> None:
        super(Spectrogram, self).__init__()
        self.n_fft = n_fft
        # number of FFT bins. the returned STFT result will have n_fft // 2 + 1
        # number of frequecies due to onesided=True in torch.stft
        self.win_length = win_length if win_length is not None else n_fft
        self.hop_length = hop_length if hop_length is not None else self.win_length // 2
        window = window_fn(self.win_length) if wkwargs is None else window_fn(self.win_length, **wkwargs)
        self.register_buffer("window", window)
        self.pad = pad
        self.power = power
        self.normalized = normalized

    def forward(self, waveform: Tensor) -> Tensor:
        """Compute the spectrogram of the input waveform.

        Returns:
            Spectrogram tensor of shape (..., freq, time), where freq is
            ``n_fft // 2 + 1`` and time is the number of window hops.
        """
        return spectrogram(
            waveform, self.pad, self.window, self.n_fft, self.hop_length, self.win_length, self.power, self.normalized
        )


@delegates(Spectrogram, keep=True)
class Sequence2Spectrogram(Transform):
    """Transform that computes a spectrogram from a time series sequence.

    Args:
        scaling: amplitude scaling mode ('log' for log10).
    """

    def __init__(self, scaling: str = "log", **kwargs):
        self.scaling = scaling
        self.tfm = Spectrogram(**kwargs)

    def encodes(self, o: TensorSpectrogram):
        if o.device != self.tfm.window.device:
            self.tfm.window = self.tfm.window.to(o.device)
            #         import pdb;pdb.set_trace()
        spec = self.tfm(o.transpose(-1, -2).contiguous())
        if self.scaling == "log":
            spec = torch.log10(spec + 1e-10)
        return spec


class SpectrogramBlock(TransformBlock):
    """TransformBlock that converts sequences to spectrograms via STFT.

    Args:
        seq_extract: transform that extracts the raw sequence.
        padding: whether to pad sequences of different lengths.
        n_fft: FFT size for the spectrogram transform.
        hop_length: hop between STFT windows, or None for default.
        normalized: whether to normalize the STFT output.
    """

    def __init__(
        self,
        seq_extract: Transform,
        padding: bool = False,
        n_fft: int = 100,
        hop_length: int | None = None,
        normalized: bool = False,
    ):
        return super().__init__(
            type_tfms=[seq_extract],
            batch_tfms=[Sequence2Spectrogram(n_fft=n_fft, hop_length=hop_length, normalized=normalized)],
            dls_kwargs={} if not padding else {"before_batch": pad_sequence},
        )

    @classmethod
    @delegates(HDF2Sequence, keep=True)
    def from_hdf(
        cls,
        clm_names: list[str],
        seq_cls: type = TensorSpectrogramInput,
        padding: bool = False,
        n_fft: int = 100,
        hop_length: int | None = None,
        normalized: bool = False,
        **kwargs,
    ):
        """Create a SpectrogramBlock from HDF5 files.

        Args:
            clm_names: column/dataset names to extract from the HDF5 file.
            seq_cls: tensor class for the extracted sequences.
            padding: whether to pad sequences of different lengths.
            n_fft: FFT size for the spectrogram transform.
            hop_length: hop between STFT windows, or None for default.
            normalized: whether to normalize the STFT output.
        """
        return cls(
            HDF2Sequence(clm_names, to_cls=seq_cls, **kwargs),
            padding,
            n_fft=n_fft,
            hop_length=hop_length,
            normalized=normalized,
        )
