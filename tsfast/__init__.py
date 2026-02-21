__version__ = "0.2.0"

# Disable MPS auto-selection: PyTorch MPS backend is often unstable or slow for
# time series workloads. Set TSFAST_USE_MPS=1 to opt in to MPS device selection.
import os, torch, fastai.torch_core as _ftc
_ftc._has_mps = lambda: (
    os.environ.get('TSFAST_USE_MPS') == '1'
    and hasattr(torch.backends, 'mps')
    and torch.backends.mps.is_available()
)
