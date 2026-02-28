"""Spectrogram transforms for frequency-domain analysis."""

__all__ = [
    "Spectrogram",
    "spectrogram",
    "complex_norm",
]

from collections.abc import Callable

import torch
from torch import Tensor


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
