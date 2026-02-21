__all__ = ["valid_clm_splitter", "ParentSplitter", "PercentageSplitter", "ApplyToDict"]

from fastai.data.all import *


def _parent_idxs(items, name):
    return mask2idxs(Path(o).parent.name == name for o in items)


def ParentSplitter(train_name="train", valid_name="valid"):
    "Split `items` from the parent folder names (`train_name` and `valid_name`)."

    def _inner(o, **kwargs):
        # if dictionaries are provided, extract the path
        if isinstance(o[0], dict):
            o = [d["path"] for d in o]
        return _parent_idxs(o, train_name), _parent_idxs(o, valid_name)

    return _inner


def PercentageSplitter(pct=0.8):
    "Split `items` in order in relative quantity."

    def _inner(o, **kwargs):
        split_idx = int(len(o) * pct)
        return L(range(split_idx)), L(range(split_idx, len(o)))

    return _inner


def ApplyToDict(fn, key="path"):
    return lambda x: fn([i[key] for i in x])


valid_clm_splitter = FuncSplitter(lambda o: o["valid"])
