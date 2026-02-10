__version__ = "0.1.8"

# Fix fastai >=2.8.6 _has_mps() bug: falls back to is_built() when is_available()
# returns False, selecting MPS device on systems where it's not actually usable.
import torch, fastai.torch_core as _ftc
_ftc._has_mps = lambda: hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
