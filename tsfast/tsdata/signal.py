"""Signal processing utilities for resampling and downsampling time series."""

import numpy as np
import torch
from scipy.signal import butter, lfilter, lfilter_zi


def running_mean(x: np.ndarray, N: int) -> np.ndarray:
    """Compute running mean with window size N."""
    cumsum = np.cumsum(np.insert(x, 0, 0, axis=0), axis=0)
    return (cumsum[N:] - cumsum[:-N]) / float(N)


def downsample_mean(x: np.ndarray, N: int) -> np.ndarray:
    """Downsample by averaging consecutive groups of N samples."""
    trunc = -(x.shape[0] % N)
    trunc = trunc if trunc != 0 else None
    return x[:trunc, :].reshape((-1, N, x.shape[-1])).mean(axis=1)


def resample_interp(
    x: np.ndarray,
    resampling_factor: float,
    sequence_first: bool = True,
    lowpass_cut: float = 1.0,
    upsample_cubic_cut: float | None = None,
) -> np.ndarray:
    """Signal resampling using linear or cubic interpolation.

    Args:
        x: signal to resample with shape features x resampling_dimension
            or resampling_dimension x features if sequence_first=True
        resampling_factor: factor > 0 that scales the signal
        sequence_first: whether the resampling dimension is the first axis
        lowpass_cut: upper boundary for resampling_factor that activates the
            lowpass filter, low values exchange accuracy for performance
        upsample_cubic_cut: lower boundary for resampling_factor that activates
            cubic interpolation at high upsampling values, None deactivates it
    """
    if sequence_first:
        x = x.T

    fs_n = resampling_factor
    # if downsampling rate is too high, lowpass filter before interpolation
    if fs_n < lowpass_cut:
        b, a = butter(2, fs_n)
        zi = lfilter_zi(b, a) * x[:, :1]
        x, _ = lfilter(b, a, x, axis=-1, zi=zi)

    x_int = torch.tensor(x, dtype=torch.float64)[None, ...]
    targ_size = int(x.shape[-1] * fs_n)

    if upsample_cubic_cut is None or fs_n <= upsample_cubic_cut:
        x = torch.nn.functional.interpolate(x_int, size=targ_size, mode="linear", align_corners=False)[0].numpy()
    else:
        x = torch.nn.functional.interpolate(x_int[..., None], size=[targ_size, 1], mode="bicubic", align_corners=False)[
            0, ..., 0
        ].numpy()

    if sequence_first:
        x = x.T

    return x
