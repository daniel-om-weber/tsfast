"""CUDA-graphed wrapper for stateful models."""

import gc
import warnings

import torch
from torch import Tensor, nn

from .state import StateSpec, discover_state_spec, flatten_state, unflatten_state


class _ThreadLocalCaptureGraph(torch.cuda.graphs.graph):
    """``torch.cuda.graph`` with ``capture_error_mode`` forced to ``"thread_local"``.

    ``make_graphed_callables`` captures under the default ``"global"`` mode, in which CUDA
    polices every thread of the process: a potentially-unsafe call from *any* of them —
    above all ``cudaHostAlloc``, which a ``DataLoader(pin_memory=True)`` issues from
    whichever thread fetches the batch (tsfast's ``PrefetchLoader`` producer, or torch's
    own pin-memory thread when ``num_workers > 0``) — fails there with
    ``cudaErrorStreamCaptureUnsupported`` *and* invalidates the in-flight capture, which
    then dies in ``capture_end`` with ``cudaErrorStreamCaptureInvalidated``. Whether the
    two collide is a timing race against the loader, so the failure is intermittent.

    A capture only records work the capturing thread submits to its own stream, so
    ``"thread_local"`` is the accurate scope, not a workaround — it is also what torch's
    inductor CUDA-graph backend uses (``cudagraph_trees``). ``make_graphed_callables``
    exposes no parameter for it (torch 2.12), so ``_init_graph`` temporarily rebinds
    ``torch.cuda.graph`` to this subclass for the duration of that one call.
    """

    def __init__(self, cuda_graph, pool=None, stream=None, capture_error_mode="global"):
        super().__init__(cuda_graph, pool=pool, stream=stream, capture_error_mode="thread_local")


class _FlatStateBridge(nn.Module):
    """Wraps a stateful model so state is passed as a flat ``[B, D]`` tensor.

    Required for ``make_graphed_callables`` which only accepts Tensor arguments.
    """

    def __init__(self, model: nn.Module, spec: StateSpec):
        super().__init__()
        self.model = model
        self._spec = spec

    def forward(self, x: Tensor, flat_state: Tensor) -> tuple[Tensor, Tensor]:
        state = unflatten_state(flat_state, self._spec)
        pred, new_state = self.model(x, state=state)
        new_flat = flatten_state(new_state, batch_size=x.shape[0])
        return pred, new_flat


class GraphedStatefulModel(nn.Module):
    """Wraps a stateful model with CUDA-graphed forward, same interface.

    The model must return ``(output, state)`` from ``forward()``.
    The CUDA graph is captured lazily on the first forward call.
    When input shapes change (e.g. different batch size at test time),
    falls back to eager execution automatically. A capture that fails outright
    also falls back to eager — permanently, with a ``RuntimeWarning`` naming the
    error — after restoring the CUDA state a broken capture leaves behind
    (current stream and RNG; see ``_recover_from_failed_capture``).

    Args:
        model: stateful model returning ``(output, state)``
        num_warmup_iters: warmup iterations before graph capture
    """

    def __init__(self, model: nn.Module, num_warmup_iters: int = 3):
        super().__init__()
        self.model = model
        self.num_warmup_iters = num_warmup_iters
        self._graphed = None
        self._spec: StateSpec | None = None
        self._zero_flat: Tensor | None = None
        self._graphed_shape: tuple[int, ...] | None = None
        self._capture_failed = False

    def reset_graph(self):
        """Clear captured graph (and any capture failure) for re-capture on next forward call."""
        self._graphed = None
        self._spec = None
        self._zero_flat = None
        self._graphed_shape = None
        self._capture_failed = False

    def _init_graph(self, x: Tensor):
        device = x.device
        assert device.type == "cuda", "GraphedStatefulModel requires a CUDA device"
        spec = discover_state_spec(self.model, x.shape[-1], device)
        self._spec = spec
        dtype = next(self.model.parameters()).dtype
        self._zero_flat = torch.zeros(x.shape[0], spec.state_size, device=device, dtype=dtype)
        wrapper = _FlatStateBridge(self.model, spec)
        sample_x = torch.zeros_like(x)
        sample_state = torch.zeros_like(self._zero_flat)
        # Graphs discarded earlier stay alive in reference cycles until a cyclic
        # collection frees them. CUDA forbids destroying a graph while a capture is in
        # flight and invalidates that capture, so drop them first and keep the collector
        # from running inside the capture below.
        gc.collect()
        gc_was_enabled = gc.isenabled()
        gc.disable()
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", "The AccumulateGrad node's stream")
                orig_graph = torch.cuda.graph
                torch.cuda.graph = _ThreadLocalCaptureGraph
                try:
                    self._graphed = torch.cuda.make_graphed_callables(
                        wrapper, (sample_x, sample_state), num_warmup_iters=self.num_warmup_iters
                    )
                except RuntimeError as err:  # torch.AcceleratorError is a RuntimeError
                    self._recover_from_failed_capture(device, err)
                finally:
                    torch.cuda.graph = orig_graph
        finally:
            if gc_was_enabled:
                gc.enable()
        if self._graphed is not None:
            self._graphed_shape = tuple(x.shape)

    def _recover_from_failed_capture(self, device, err: RuntimeError):
        """Restore CUDA state after a failed capture, then fall back to eager — loudly.

        A capture that dies in ``capture_end`` leaves two process-wide landmines behind:

        * ``torch.cuda.graph.__exit__`` calls ``capture_end`` *before* popping its stream
          context, so on failure the thread stays stuck on the capture side stream;
        * ``CUDAGraph::capture_end`` raises before running the RNG ``capture_epilogue``,
          so the default CUDA generator stays in capture mode and every later RNG op
          (dropout, ``torch.randn`` in an augmentation, ...) fails with
          "Offset increment outside graph capture encountered unexpectedly."

        The stream is restored directly; the generator is repaired by running one
        trivial capture to completion, whose epilogue clears the capture flag. If the
        repair does not hold, raise instead of limping on with poisoned RNG.
        """
        self._capture_failed = True
        torch.cuda.set_stream(torch.cuda.default_stream(device))
        torch.cuda.synchronize()
        try:
            torch.rand(1, device=device)
            torch.cuda.synchronize()
        except RuntimeError:
            repair = torch.cuda.CUDAGraph()
            with _ThreadLocalCaptureGraph(repair):
                torch.zeros(1, device=device).add_(1)
            del repair
            try:
                torch.rand(1, device=device)
                torch.cuda.synchronize()
            except RuntimeError as rng_err:
                raise RuntimeError(
                    "CUDA graph capture failed and left the CUDA RNG in capture mode; "
                    "recovery failed, so this process cannot run further CUDA RNG ops. "
                    f"Original capture error: {err}"
                ) from rng_err
        warnings.warn(
            f"CUDA graph capture failed; {type(self.model).__name__} falls back to eager "
            "execution for the rest of training, losing the cuda_graph speedup. Captures "
            "run in thread_local mode, so other threads (e.g. DataLoader pinning memory) "
            "cannot be the cause. Either the model does something capture-unsafe (CPU "
            "synchronization, dynamic shapes, an unsupported op), or outputs of an earlier "
            "grad-enabled forward of this model are still alive — their autograd graph ties "
            "the parameters' AccumulateGrad nodes to the legacy default stream, which the "
            "captured backward is not allowed to depend on (cudaErrorStreamCaptureImplicit); "
            "delete or .detach() those outputs before the first training step. Pass "
            f"cuda_graph=False to silence this warning. Capture error: {err}",
            RuntimeWarning,
            stacklevel=4,
        )

    def forward(self, x: Tensor, state=None) -> tuple[Tensor, ...]:
        if self._graphed is None and not self._capture_failed:
            self._init_graph(x)
        if self._graphed is None or tuple(x.shape) != self._graphed_shape:
            return self.model(x, state=state)
        if state is not None:
            flat_state = flatten_state(state, batch_size=x.shape[0])
        else:
            flat_state = torch.zeros_like(self._zero_flat)
        pred, new_flat = self._graphed(x, flat_state)
        new_state = unflatten_state(new_flat, self._spec)
        return pred, new_state
