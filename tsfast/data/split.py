"""Train/validation splitting strategies for data pipelines."""

__all__ = ["valid_clm_splitter", "ParentSplitter", "PercentageSplitter", "ApplyToDict"]

from fastai.data.all import *


def _parent_idxs(items, name):
    return mask2idxs(Path(o).parent.name == name for o in items)


def ParentSplitter(train_name="train", valid_name="valid"):
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


def PercentageSplitter(pct=0.8):
    """Split items sequentially by a percentage threshold.

    Args:
        pct: fraction of items to assign to the training set.
    """

    def _inner(o, **kwargs):
        split_idx = int(len(o) * pct)
        return L(range(split_idx)), L(range(split_idx, len(o)))

    return _inner


def ApplyToDict(fn, key="path"):
    """Wrap a splitter function to operate on a specific dict key.

    Args:
        fn: splitter function to apply on the extracted list.
        key: dictionary key to extract from each item.
    """
    return lambda x: fn([i[key] for i in x])


valid_clm_splitter = FuncSplitter(lambda o: o["valid"])
