"""Data augmentation transforms for time series sequences."""

__all__ = [
    "SeqSlice",
    "SeqNoiseInjection",
    "SeqNoiseInjection_Varying",
    "SeqNoiseInjection_Grouped",
    "SeqBiasInjection",
]

from fastai.basics import *
from .core import TensorSequencesInput
from fastai.vision.augment import RandTransform


class SeqSlice(Transform):
    """Slice a subsequence from an array-like object.

    Args:
        l_slc: left slice boundary index.
        r_slc: right slice boundary index.
    """

    def __init__(self, l_slc=None, r_slc=None):
        self.l_slc, self.r_slc = l_slc, r_slc

    def encodes(self, o):
        return o[self.l_slc : self.r_slc]


class SeqNoiseInjection(RandTransform):
    """Add normal-distributed noise with per-signal mean and std.

    Only applied to training data.

    Args:
        std: standard deviation of the noise per signal.
        mean: mean of the noise per signal.
        p: probability of applying the transform.
    """

    split_idx = 0

    def __init__(self, std=1e-1, mean=0.0, p=1.0):
        super().__init__(p=p)
        self.std = tensor(std).type(torch.float)
        self.mean = tensor(mean).type(torch.float)

    def encodes(self, o: TensorSequencesInput):
        if o.device != self.mean.device:
            self.std = self.std.to(o.device)
            self.mean = self.mean.to(o.device)
        # expand creates a view on a tensor and is therefore very fast compared to copy
        return o + torch.normal(mean=self.mean.expand_as(o), std=self.std.expand_as(o))


class SeqNoiseInjection_Varying(RandTransform):
    """Add noise with a randomly sampled standard deviation per application.

    Only applied to training data.

    Args:
        std_std: standard deviation of the noise std distribution.
        p: probability of applying the transform.
    """

    split_idx = 0

    def __init__(self, std_std=0.1, p=1.0):
        super().__init__(p=p)
        self.std_std = tensor(std_std).type(torch.float)

    def encodes(self, o: TensorSequencesInput):
        if o.device != self.std_std.device:
            self.std_std = self.std_std.to(o.device)

        # expand creates a view on a tensor and is therefore very fast compared to copy
        std = torch.normal(mean=0, std=self.std_std).abs()
        return o + torch.normal(mean=0, std=std.expand_as(o))


class SeqNoiseInjection_Grouped(RandTransform):
    """Add noise with per-group randomly sampled standard deviations.

    Only applied to training data. Each group of signals shares a
    randomly drawn noise std.

    Args:
        std_std: standard deviation of the noise std distribution per group.
        std_idx: index mapping each signal to its noise group.
        p: probability of applying the transform.
    """

    split_idx = 0

    def __init__(self, std_std, std_idx, p=1.0):
        super().__init__(p=p)
        self.std_std = tensor(std_std).type(torch.float)
        self.std_idx = tensor(std_idx).type(torch.long)

    def encodes(self, o: TensorSequencesInput):
        if o.device != self.std_std.device:
            self.std_std = self.std_std.to(o.device)

        # expand creates a view on a tensor and is therefore very fast compared to copy
        std = torch.normal(mean=0, std=self.std_std).abs()[self.std_idx]
        return o + torch.normal(mean=0, std=std.expand_as(o))


class SeqBiasInjection(RandTransform):
    """Add a constant normal-distributed offset per signal per sample.

    Only applied to training data.

    Args:
        std: standard deviation of the bias per signal.
        mean: mean of the bias per signal.
        p: probability of applying the transform.
    """

    split_idx = 0

    def __init__(self, std=1e-1, mean=0.0, p=1.0):
        super().__init__(p=p)
        self.std = tensor(std).type(torch.float)
        self.mean = tensor(mean).type(torch.float)

    def encodes(self, o: TensorSequencesInput):
        if o.device != self.mean.device:
            self.std = self.std.to(o.device)
            self.mean = self.mean.to(o.device)

        # expand creates a view on a tensor and is therefore very fast compared to copy
        mean = self.mean.repeat((o.shape[0], 1, 1)).expand((o.shape[0], 1, o.shape[2]))
        std = self.std.repeat((o.shape[0], 1, 1)).expand((o.shape[0], 1, o.shape[2]))
        n = torch.normal(mean=mean, std=std).expand_as(o)
        return o + n
