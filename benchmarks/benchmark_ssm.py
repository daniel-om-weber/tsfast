#!/usr/bin/env python
"""Benchmark: NeuralStateSpace execution backends (eager vs compiled vs c vs triton).

Times a full training step (forward + MSE + backward + Adam) and inference of the
sequential rollout, reported as microseconds per trajectory. Runs on synthetic
data — no dataset download required.

Usage:
    uv run python benchmarks/benchmark_ssm.py
    uv run python benchmarks/benchmark_ssm.py --device cpu --batch-sizes 8 32
    uv run python benchmarks/benchmark_ssm.py --hidden 128 128 --seq-len 500
"""

import argparse
import time

import torch
import torch.nn.functional as F

from tsfast.models.ssm import NeuralStateSpace

N_STATE = 10
N_INPUT = 10
N_WARMUP = 10
N_TIMED = 30
SEED = 42


def detect_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sync(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def bench(fn, device, n_warmup=N_WARMUP, n_timed=N_TIMED) -> float:
    """Median-free simple timing: mean ms over n_timed iterations."""
    for _ in range(n_warmup):
        fn()
    sync(device)
    t0 = time.perf_counter()
    for _ in range(n_timed):
        fn()
    sync(device)
    return (time.perf_counter() - t0) / n_timed * 1e3


def backends_for(device: torch.device, include_compiled: bool) -> list[str]:
    names = ["eager"]
    if include_compiled:
        names.append("compiled")
    if device.type == "cuda":
        from tsfast.models.ssm import backend_triton as ssm_triton

        if ssm_triton.is_available():
            names.append("triton")
    else:
        from tsfast.models.ssm import backend_c as ssm_c

        if ssm_c.is_available():
            names.append("c")
    return names


def run(args):
    device = torch.device(args.device) if args.device else detect_device()
    torch.manual_seed(SEED)
    hidden = list(args.hidden)
    names = backends_for(device, args.compiled)
    print(f"device={device.type}  hidden={hidden}  L={args.seq_len}  (us per trajectory)")
    header = f"{'train step':>12s}" + "".join(f"{'B=' + str(b):>12s}" for b in args.batch_sizes)
    print(header)
    print("-" * len(header))
    for name in names:
        cells = []
        for B in args.batch_sizes:
            m = NeuralStateSpace(N_STATE, N_INPUT, hidden, backend=name).to(device)
            u = torch.randn(B, args.seq_len, N_INPUT, device=device)
            x0 = torch.zeros(B, N_STATE, device=device)
            tgt = torch.randn(B, args.seq_len, N_STATE, device=device)
            opt = torch.optim.Adam(m.parameters(), lr=1e-3)

            def step():
                opt.zero_grad(set_to_none=True)
                loss = F.mse_loss(m(u, x0), tgt)
                loss.backward()
                opt.step()

            cells.append(bench(step, device) / B * 1e3)
        print(f"{name:>12s}" + "".join(f"{c:>12.2f}" for c in cells))

    print(f"\n{'inference':>12s}" + "".join(f"{'B=' + str(b):>12s}" for b in args.batch_sizes))
    print("-" * len(header))
    for name in names:
        cells = []
        for B in args.batch_sizes:
            m = NeuralStateSpace(N_STATE, N_INPUT, hidden, backend=name).to(device).eval()
            u = torch.randn(B, args.seq_len, N_INPUT, device=device)
            x0 = torch.zeros(B, N_STATE, device=device)
            with torch.no_grad():
                cells.append(bench(lambda: m(u, x0), device) / B * 1e3)
        print(f"{name:>12s}" + "".join(f"{c:>12.2f}" for c in cells))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", default=None, help="cuda or cpu (auto-detected if omitted)")
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[16, 64, 256])
    p.add_argument("--seq-len", type=int, default=300)
    p.add_argument("--hidden", type=int, nargs="+", default=[64, 64])
    p.add_argument("--compiled", action="store_true", help="include the torch.compile backend (slow first call)")
    run(p.parse_args())


if __name__ == "__main__":
    main()
