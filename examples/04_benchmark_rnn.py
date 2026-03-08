#!/usr/bin/env python
"""Benchmark: torch.compile vs CUDA Graphs for RNN training acceleration.

Compares wall-clock training speed of different acceleration strategies
for GRU-based TBPTT training.  Runs on synthetic data — no dataset
download required.

Usage:
    uv run python examples/04_benchmark_rnn.py
"""

import time

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from tsfast.models.layers import SeqLinear
from tsfast.models.rnn import SimpleRNN
from tsfast.training import TbpttLearner
from tsfast.models.state import GraphedStatefulModel, detach_state

# ── Configuration ────────────────────────────────────────────────────────────

N_U, N_Y = 1, 1
HIDDEN_SIZE = 200
NUM_LAYERS = 2
RNN_TYPE = "gru"
WIN_SZ = 1800
SUB_SEQ_LEN = 200  # 1800 / 200 = 9 chunks per batch
BS = 16
N_TRAIN = 128
N_VALID = 32
LR = 3e-3
BENCH_SECONDS = 10  # time budget per method (after warmup)
N_WARMUP = 2  # warmup epochs (compilation overhead absorbed here)
SEED = 42

# ── Synthetic Data ───────────────────────────────────────────────────────────


class BenchmarkDls:
    """Minimal DataLoaders for benchmarking — no HDF5 dependency."""

    def __init__(self):
        torch.manual_seed(0)
        self.train = DataLoader(
            TensorDataset(
                torch.randn(N_TRAIN, WIN_SZ, N_U),
                torch.randn(N_TRAIN, WIN_SZ, N_Y),
            ),
            batch_size=BS,
            shuffle=False,
        )
        self.valid = DataLoader(
            TensorDataset(
                torch.randn(N_VALID, WIN_SZ, N_U),
                torch.randn(N_VALID, WIN_SZ, N_Y),
            ),
            batch_size=BS,
        )
        self.test = None


# ── Model Factory ────────────────────────────────────────────────────────────


def make_model():
    torch.manual_seed(SEED)
    return SimpleRNN(
        N_U, N_Y,
        num_layers=NUM_LAYERS,
        hidden_size=HIDDEN_SIZE,
        rnn_type=RNN_TYPE,
        return_state=True,
    )


# ── Custom GRU (pure PyTorch, no cuDNN — torch.compile can fully trace) ──────


class CustomGRUCell(nn.Module):
    """GRU cell implemented with linear layers — fully traceable by torch.compile."""

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.W_ir = nn.Linear(input_size, hidden_size)
        self.W_hr = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_iz = nn.Linear(input_size, hidden_size)
        self.W_hz = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_in = nn.Linear(input_size, hidden_size)
        self.W_hn = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x: Tensor, h: Tensor) -> Tensor:
        r = torch.sigmoid(self.W_ir(x) + self.W_hr(h))
        z = torch.sigmoid(self.W_iz(x) + self.W_hz(h))
        n = torch.tanh(self.W_in(x) + r * self.W_hn(h))
        return (1 - z) * n + z * h


class CustomGRULayer(nn.Module):
    """Single-layer GRU that loops over timesteps using CustomGRUCell."""

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = CustomGRUCell(input_size, hidden_size)

    def forward(self, x: Tensor, h: Tensor | None = None) -> tuple[Tensor, Tensor]:
        bs, seq_len, _ = x.shape
        if h is None:
            h = torch.zeros(1, bs, self.hidden_size, device=x.device, dtype=x.dtype)
        h = h.squeeze(0)  # [bs, hidden]
        outputs = []
        for t in range(seq_len):
            h = self.cell(x[:, t], h)
            outputs.append(h)
        return torch.stack(outputs, dim=1), h.unsqueeze(0)


class CustomGRUModel(nn.Module):
    """Multi-layer custom GRU + linear head, matching SimpleRNN's interface."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        num_layers: int = 1,
        hidden_size: int = 100,
        return_state: bool = False,
    ):
        super().__init__()
        self.return_state = return_state
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(CustomGRULayer(input_size if i == 0 else hidden_size, hidden_size))
        self.final = SeqLinear(hidden_size, output_size, hidden_size=hidden_size, hidden_layer=0)

    def forward(self, x: Tensor, state: list | None = None):
        new_hidden = []
        for i, layer in enumerate(self.layers):
            h = state[i] if state is not None else None
            x, h_new = layer(x, h)
            new_hidden.append(h_new)
        out = self.final(x)
        return out if not self.return_state else (out, new_hidden)


def make_custom_model():
    torch.manual_seed(SEED)
    return CustomGRUModel(
        N_U, N_Y,
        num_layers=NUM_LAYERS,
        hidden_size=HIDDEN_SIZE,
        return_state=True,
    )


# ── Compiled-Step Learner ────────────────────────────────────────────────────


class CompiledStepTbpttLearner(TbpttLearner):
    """TbpttLearner that compiles the forward+backward step with torch.compile.

    The optimizer step stays outside the compiled region (same as
    GraphedStatefulModel) so LR schedulers and grad clipping work normally.
    """

    def __init__(self, *args, compile_mode: str = "reduce-overhead", **kwargs):
        super().__init__(*args, **kwargs)
        self._compile_mode = compile_mode
        self._compiled_fwd_bwd: dict = {}  # keyed by (has_state, n_skip)

    def _get_compiled_fn(self, has_state: bool, n_skip: int):
        key = (has_state, n_skip)
        if key in self._compiled_fwd_bwd:
            return self._compiled_fwd_bwd[key]

        model = self.model
        loss_func = self.loss_func

        if has_state and n_skip > 0:
            def _fwd_bwd(xb: Tensor, yb: Tensor, state: list) -> tuple[Tensor, list]:
                pred, new_state = model(xb, state=state)
                loss = loss_func(pred[:, n_skip:], yb[:, n_skip:])
                loss.backward()
                return loss, new_state
        elif has_state:
            def _fwd_bwd(xb: Tensor, yb: Tensor, state: list) -> tuple[Tensor, list]:
                pred, new_state = model(xb, state=state)
                loss = loss_func(pred, yb)
                loss.backward()
                return loss, new_state
        elif n_skip > 0:
            def _fwd_bwd(xb: Tensor, yb: Tensor) -> tuple[Tensor, list]:
                pred, new_state = model(xb)
                loss = loss_func(pred[:, n_skip:], yb[:, n_skip:])
                loss.backward()
                return loss, new_state
        else:
            def _fwd_bwd(xb: Tensor, yb: Tensor) -> tuple[Tensor, list]:
                pred, new_state = model(xb)
                loss = loss_func(pred, yb)
                loss.backward()
                return loss, new_state

        compiled = torch.compile(_fwd_bwd, mode=self._compile_mode)
        self._compiled_fwd_bwd[key] = compiled
        return compiled

    def _forward_backward_step(
        self, xb: Tensor, yb: Tensor, optimizer, state=None, n_skip: int | None = None
    ) -> tuple[float | None, object]:
        if n_skip is None:
            n_skip = self.n_skip

        has_state = state is not None
        fn = self._get_compiled_fn(has_state, n_skip)

        if has_state:
            loss, new_state = fn(xb, yb, state)
        else:
            loss, new_state = fn(xb, yb)

        if torch.isnan(loss):
            optimizer.zero_grad()
            return None, None

        if self.grad_clip is not None:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        optimizer.step()
        optimizer.zero_grad()
        return loss.item(), detach_state(new_state)


# ── Benchmark Harness ────────────────────────────────────────────────────────


def run_benchmark(
    learner: TbpttLearner, n_warmup: int = N_WARMUP, bench_seconds: float = BENCH_SECONDS
) -> tuple[list[float], float]:
    """Run warmup epochs, then time as many epochs as fit in bench_seconds.

    Returns:
        (epoch_times_seconds, final_val_loss)
    """
    learner.model.to(learner.device)
    optimizer = torch.optim.Adam(learner.model.parameters(), lr=LR)
    # Use a large total_steps estimate so pct_train stays reasonable
    total_steps = 1_000_000
    step = 0

    # Warmup epochs (compilation / graph capture happens here)
    for _ in range(n_warmup):
        learner.model.train()
        for batch in learner.dls.train:
            learner._train_one_batch(batch, optimizer, step, total_steps)
            step += 1

    # Timed epochs — run until time budget exhausted
    epoch_times: list[float] = []
    deadline = time.perf_counter() + bench_seconds
    while time.perf_counter() < deadline:
        learner.model.train()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for batch in learner.dls.train:
            learner._train_one_batch(batch, optimizer, step, total_steps)
            step += 1
        torch.cuda.synchronize()
        epoch_times.append(time.perf_counter() - t0)

    val_loss, _ = learner.validate()
    return epoch_times, val_loss


# ── Method Registry ──────────────────────────────────────────────────────────

METHODS: dict[str, tuple] = {}  # name -> (factory_fn, description)


def register(name: str, desc: str):
    def decorator(fn):
        METHODS[name] = (fn, desc)
        return fn
    return decorator


@register("baseline", "nn.GRU + TbpttLearner (baseline)")
def _baseline(dls):
    return TbpttLearner(make_model(), dls, loss_func=nn.L1Loss(), sub_seq_len=SUB_SEQ_LEN)


@register("cuda_graph", "nn.GRU + GraphedStatefulModel")
def _cuda_graph(dls):
    model = GraphedStatefulModel(make_model())
    return TbpttLearner(model, dls, loss_func=nn.L1Loss(), sub_seq_len=SUB_SEQ_LEN)


@register("custom_compile_reduce", "custom GRU + compile (reduce-overhead)")
def _custom_compile_reduce(dls):
    model = torch.compile(make_custom_model(), mode="reduce-overhead")
    return TbpttLearner(model, dls, loss_func=nn.L1Loss(), sub_seq_len=SUB_SEQ_LEN)


@register("custom_compile_autotune", "custom GRU + compile (max-autotune)")
def _custom_compile_autotune(dls):
    model = torch.compile(make_custom_model(), mode="max-autotune")
    return TbpttLearner(model, dls, loss_func=nn.L1Loss(), sub_seq_len=SUB_SEQ_LEN)


@register("custom_compile_step", "custom GRU + compile step (reduce-overhead)")
def _custom_compile_step(dls):
    return CompiledStepTbpttLearner(
        make_custom_model(), dls,
        loss_func=nn.L1Loss(),
        sub_seq_len=SUB_SEQ_LEN,
        compile_mode="reduce-overhead",
    )


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    assert torch.cuda.is_available(), "This benchmark requires CUDA"

    print("=== RNN Training Benchmark: torch.compile vs CUDA Graphs ===")
    print(f"Config: {RNN_TYPE.upper()}, {NUM_LAYERS} layers, {HIDDEN_SIZE} hidden, "
          f"{WIN_SZ} window, sub_seq={SUB_SEQ_LEN}, bs={BS}")
    print(f"Data:   {N_TRAIN} train / {N_VALID} valid samples (synthetic)")
    print(f"Timing: {N_WARMUP} warmup epochs + {BENCH_SECONDS}s timed per method")
    print()

    results: dict[str, dict] = {}
    for name, (factory, desc) in METHODS.items():
        print(f"  {desc} ...", end=" ", flush=True)
        torch._dynamo.reset()
        dls = BenchmarkDls()
        learner = factory(dls)

        try:
            epoch_times, val_loss = run_benchmark(learner)
            mean_ms = 1000 * sum(epoch_times) / len(epoch_times)
            results[name] = {
                "desc": desc,
                "epoch_times": epoch_times,
                "mean_ms": mean_ms,
                "val_loss": val_loss,
            }
            print(f"{mean_ms:.0f} ms/epoch ({len(epoch_times)} epochs)")
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            results[name] = {"desc": desc, "error": str(e)}
            print(f"FAILED: {e}")

        del learner
        torch.cuda.empty_cache()

    # ── Results table ────────────────────────────────────────────────────
    print()
    baseline_ms = results.get("baseline", {}).get("mean_ms", 1.0)
    header = f"{'Method':<40} | {'Epoch (ms)':>10} | {'Speedup':>7} | {'Val Loss':>8}"
    print(header)
    print("-" * len(header))
    for name, res in results.items():
        if "error" in res:
            print(f"{res['desc']:<40} | {'ERROR':>10} | {'---':>7} | {'---':>8}")
        else:
            speedup = baseline_ms / res["mean_ms"]
            print(f"{res['desc']:<40} | {res['mean_ms']:>10.1f} | {speedup:>6.2f}x | {res['val_loss']:>8.4f}")

    # ── Correctness check ────────────────────────────────────────────────
    print()
    baseline_loss = results.get("baseline", {}).get("val_loss")
    if baseline_loss is not None:
        print("Correctness vs baseline:")
        for name, res in results.items():
            if "error" in res:
                print(f"  {res['desc']}: SKIPPED (error)")
            elif name == "baseline":
                continue
            else:
                diff = abs(res["val_loss"] - baseline_loss)
                status = "OK" if diff < 0.05 else "WARN"
                print(f"  {res['desc']}: diff={diff:.6f} [{status}]")



if __name__ == "__main__":
    main()
