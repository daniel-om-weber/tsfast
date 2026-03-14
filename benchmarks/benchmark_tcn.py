#!/usr/bin/env python
"""Benchmark: torch.compile effect on TCN forward+backward performance.

Compares eager vs compiled TCN across model sizes and sequence lengths.
Runs on synthetic data — no dataset download required.  Auto-detects
the best available device (CUDA > MPS > CPU).

Usage:
    uv run python benchmarks/benchmark_tcn.py
    uv run python benchmarks/benchmark_tcn.py --device cpu
    uv run python benchmarks/benchmark_tcn.py --device mps --seq-lens 100 500
"""

import argparse
import time

import torch

from tsfast.models.cnn import TCN

# ── Configuration ────────────────────────────────────────────────────────────

BATCH_SIZE = 16
INPUT_SIZE = 3
OUTPUT_SIZE = 1
N_WARMUP = 10
N_TIMED = 50
SEED = 42

MODEL_CONFIGS = [
    {"label": "small (d=4, w=32)", "hl_depth": 4, "hl_width": 32},
    {"label": "large (d=8, w=128)", "hl_depth": 8, "hl_width": 128},
]
SEQ_LENS = [100, 1000, 3000]


# ── Device helpers ───────────────────────────────────────────────────────────


def detect_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sync(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


# ── Benchmark core ───────────────────────────────────────────────────────────


def bench_fwd_bwd(model: torch.nn.Module, x: torch.Tensor, device: torch.device) -> tuple[float, float]:
    """Return mean forward and backward times in ms."""
    for _ in range(N_WARMUP):
        y = model(x)
        y.sum().backward()
        model.zero_grad()

    sync(device)
    fwd_times: list[float] = []
    bwd_times: list[float] = []
    for _ in range(N_TIMED):
        sync(device)
        t0 = time.perf_counter()
        y = model(x)
        sync(device)
        t1 = time.perf_counter()
        y.sum().backward()
        sync(device)
        t2 = time.perf_counter()
        model.zero_grad()
        fwd_times.append(t1 - t0)
        bwd_times.append(t2 - t1)

    fwd_ms = 1000 * sum(fwd_times) / len(fwd_times)
    bwd_ms = 1000 * sum(bwd_times) / len(bwd_times)
    return fwd_ms, bwd_ms


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Benchmark TCN: eager vs torch.compile")
    parser.add_argument("--device", type=str, default=None, help="Force device (cpu, mps, cuda)")
    parser.add_argument("--seq-lens", type=int, nargs="+", default=None, help="Sequence lengths to test")
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else detect_device()
    seq_lens = args.seq_lens or SEQ_LENS

    print(f"=== TCN Benchmark: eager vs torch.compile ({device}) ===")
    print(f"Config: batch={BATCH_SIZE}, in={INPUT_SIZE}, out={OUTPUT_SIZE}")
    print(f"Timing: {N_WARMUP} warmup + {N_TIMED} timed iterations")
    print()

    col = f"{'Model':<22} | {'Seq':>5} | {'Mode':<9} | {'Fwd (ms)':>9} | {'Bwd (ms)':>9} | {'Total':>9}"
    sep = "-" * len(col)
    print(col)
    print(sep)

    for cfg in MODEL_CONFIGS:
        for seq_len in seq_lens:
            torch.manual_seed(SEED)
            x = torch.randn(BATCH_SIZE, seq_len, INPUT_SIZE, device=device)

            # Eager
            torch._dynamo.reset()
            model_eager = TCN(INPUT_SIZE, OUTPUT_SIZE, hl_depth=cfg["hl_depth"], hl_width=cfg["hl_width"]).to(device)
            fwd_e, bwd_e = bench_fwd_bwd(model_eager, x, device)
            del model_eager

            # Compiled
            torch._dynamo.reset()
            model_compiled = torch.compile(
                TCN(INPUT_SIZE, OUTPUT_SIZE, hl_depth=cfg["hl_depth"], hl_width=cfg["hl_width"]).to(device)
            )
            try:
                fwd_c, bwd_c = bench_fwd_bwd(model_compiled, x, device)
                compiled_ok = True
            except Exception as e:
                print(f"{cfg['label']:<22} | {seq_len:>5} | compiled  | FAILED: {e}")
                compiled_ok = False
            del model_compiled

            tot_e = fwd_e + bwd_e
            print(f"{cfg['label']:<22} | {seq_len:>5} | {'eager':<9} | {fwd_e:>9.2f} | {bwd_e:>9.2f} | {tot_e:>9.2f}")

            if compiled_ok:
                tot_c = fwd_c + bwd_c
                sf = fwd_e / fwd_c if fwd_c > 0 else float("inf")
                sb = bwd_e / bwd_c if bwd_c > 0 else float("inf")
                st = tot_e / tot_c if tot_c > 0 else float("inf")
                print(f"{'':<22} | {'':<5} | {'compiled':<9} | {fwd_c:>9.2f} | {bwd_c:>9.2f} | {tot_c:>9.2f}")
                print(f"{'':<22} | {'':<5} | {'speedup':<9} | {sf:>8.2f}x | {sb:>8.2f}x | {st:>8.2f}x")
            print(sep)

            del x
            if device.type == "cuda":
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
