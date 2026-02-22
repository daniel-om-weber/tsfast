"""Data blocks for sequence and scalar inputs."""

__all__ = ["pad_sequence", "SequenceBlock", "ScalarNormalize", "ScalarBlock"]

from functools import partial

import torch
from torch import Tensor

from fastcore.meta import delegates
from fastai.data.block import TransformBlock
from fastai.data.load import DataLoader
from fastai.data.transforms import ToTensor, broadcast_vec
from fastai.torch_basics import DisplayedTransform, Transform, retain_types

from .core import (
    HDF2Sequence,
    HDF_Attrs2Scalars,
    HDF_DS2Scalars,
    TensorScalarsInput,
    TensorSequencesInput,
)


def pad_sequence(batch: list, sorting: bool = False):
    """Collate function that pads sequences of different lengths.

    Args:
        batch: list of tuples containing tensors to pad.
        sorting: whether to sort the batch by first tensor length descending.
    """
    # takes list of tuples as input, returns list of tuples
    sorted_batch = sorted(batch, key=lambda x: x[0].shape[0], reverse=True) if sorting else batch

    pad_func = partial(torch.nn.utils.rnn.pad_sequence, batch_first=True)
    padded_tensors = [pad_func([x[tup] for x in sorted_batch]) for tup in range(len(batch[0]))]
    padded_list = [retain_types(tuple([tup[entry] for tup in padded_tensors]), batch[0]) for entry in range(len(batch))]
    # retain types is important for decoding later back to source items
    #     import pdb; pdb.set_trace()

    return padded_list


class SequenceBlock(TransformBlock):
    """TransformBlock for time series sequence data.

    Args:
        seq_extract: transform that extracts the sequence from the data source.
        padding: whether to pad sequences of different lengths in each batch.
    """

    def __init__(self, seq_extract: Transform, padding: bool = False):
        return super().__init__(
            type_tfms=[seq_extract], dls_kwargs={} if not padding else {"before_batch": pad_sequence}
        )

    @classmethod
    @delegates(HDF2Sequence, keep=True)
    def from_hdf(cls, clm_names: list, seq_cls: type = TensorSequencesInput, padding: bool = False, **kwargs):
        """Create a SequenceBlock from HDF5 files.

        Args:
            clm_names: column/dataset names to extract from the HDF5 file.
            seq_cls: tensor class for the extracted sequences.
            padding: whether to pad sequences of different lengths.
        """
        return cls(HDF2Sequence(clm_names, to_cls=seq_cls, **kwargs), padding)

    @classmethod
    def from_numpy(cls, seq_cls: type = TensorSequencesInput, padding: bool = False, **kwargs):
        """Create a SequenceBlock from numpy arrays.

        Args:
            seq_cls: tensor class for the extracted sequences.
            padding: whether to pad sequences of different lengths.
        """
        return cls(ToTensor(enc=seq_cls), padding)


class ScalarNormalize(DisplayedTransform):
    """Normalize scalar inputs by mean and standard deviation.

    Args:
        mean: precomputed mean for normalization, or None to compute from data.
        std: precomputed std for normalization, or None to compute from data.
        axes: axes over which to compute statistics.
    """

    def __init__(self, mean: Tensor | None = None, std: Tensor | None = None, axes: tuple = (0,)):
        self.mean = mean
        self.std = std
        self.axes = axes

    @classmethod
    def from_stats(cls, mean: float, std: float, dim: int = 1, ndim: int = 4, cuda: bool = True):
        """Create from precomputed statistics with broadcasting.

        Args:
            mean: mean value(s) for normalization.
            std: standard deviation value(s) for normalization.
            dim: dimension index for broadcasting.
            ndim: total number of dimensions for the broadcast shape.
            cuda: whether to place tensors on GPU.
        """
        return cls(*broadcast_vec(dim, ndim, mean, std, cuda=cuda))

    def setups(self, dl: DataLoader):
        if self.mean is None or self.std is None:
            b = dl.one_batch()
            for x in b:
                if isinstance(x, TensorScalarsInput):
                    self.mean, self.std = x.mean(self.axes, keepdim=True), x.std(self.axes, keepdim=True) + 1e-7
                    return

    def encodes(self, x: TensorScalarsInput):
        if x.device != self.mean.device:
            self.mean = self.mean.to(x.device)
            self.std = self.std.to(x.device)
        return (x - self.mean) / self.std

    def decodes(self, x: TensorScalarsInput):
        if x.device != self.mean.device:
            self.mean = self.mean.to(x.device)
            self.std = self.std.to(x.device)
        return x * self.std + self.mean


class ScalarBlock(TransformBlock):
    """TransformBlock for scalar input data with automatic normalization.

    Args:
        scl_extract: transform that extracts scalars from the data source.
    """

    def __init__(self, scl_extract: Transform):
        return super().__init__(type_tfms=[scl_extract], batch_tfms=[ScalarNormalize()])

    @classmethod
    @delegates(HDF_Attrs2Scalars, keep=True)
    def from_hdf_attrs(cls, clm_names: list, scl_cls: type = TensorScalarsInput, **kwargs):
        """Create a ScalarBlock from HDF5 file attributes.

        Args:
            clm_names: attribute names to extract from the HDF5 file.
            scl_cls: tensor class for the extracted scalars.
        """
        return cls(HDF_Attrs2Scalars(clm_names, to_cls=scl_cls, **kwargs))

    @classmethod
    @delegates(HDF_DS2Scalars, keep=True)
    def from_hdf_ds(cls, clm_names: list, scl_cls: type = TensorScalarsInput, **kwargs):
        """Create a ScalarBlock from HDF5 datasets.

        Args:
            clm_names: dataset names to extract from the HDF5 file.
            scl_cls: tensor class for the extracted scalars.
        """
        return cls(HDF_DS2Scalars(clm_names, to_cls=scl_cls, **kwargs))
