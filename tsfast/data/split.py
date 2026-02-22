"""Train/validation splitting strategies for data pipelines."""

__all__ = ["valid_clm_splitter", "ParentSplitter", "PercentageSplitter", "ApplyToDict"]

from collections.abc import Callable
from pathlib import Path

from fastcore.foundation import L, mask2idxs
from fastai.data.transforms import FuncSplitter


def _parent_idxs(items, name):
    return mask2idxs(Path(o).parent.name == name for o in items)


def ParentSplitter(train_name: str = "train", valid_name: str = "valid") -> Callable:
    """Split items based on parent folder names.

    Args:
        train_name: name of the parent folder for training items.
        valid_name: name of the parent folder for validation items.
    """

    def _inner(o, **kwargs):
        if isinstance(o[0], dict):
            o = [d["path"] for d in o]
        return _parent_idxs(o, train_name), _parent_idxs(o, valid_name)

    return _inner


def PercentageSplitter(pct: float = 0.8) -> Callable:
    """Split items sequentially by a percentage threshold.

    Args:
        pct: fraction of items to assign to the training set.
    """

    def _inner(o, **kwargs):
        split_idx = int(len(o) * pct)
        return L(range(split_idx)), L(range(split_idx, len(o)))

    return _inner


def ApplyToDict(fn: Callable, key: str = "path") -> Callable:
    """Wrap a splitter function to operate on a specific dict key.

    Args:
        fn: splitter function to apply on the extracted list.
        key: dictionary key to extract from each item.
    """
    return lambda x: fn([i[key] for i in x])


valid_clm_splitter = FuncSplitter(lambda o: o["valid"])
