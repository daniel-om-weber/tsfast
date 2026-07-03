#!/usr/bin/env python
"""Benchmark: NarxMLP free-run backends (eager vs compiled vs c vs triton).

Times a full training step (forward + MSE + backward + Adam) and inference of the
sequential free-run recurrence, reported as microseconds per trajectory. Runs on
synthetic data — no dataset download required.

Usage:
    uv run python benchmarks/benchmark_narx.py
    uv run python benchmarks/benchmark_narx.py --device cpu --batch-sizes 8 32
    uv run python benchmarks/benchmark_narx.py --hidden-size 128 --seq-len 500
"""

import argparse
import time

import torch
import torch.nn.functional as F

from tsfast.models.narx import NarxMLP

N_U = 2
N_Y = 2
N_WARMUP = 10
N_TIMED = 30
SEED = 42


def detect_device() -> torch.device:
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def sync(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def bench(fn, device, n_warmup=N_WARMUP, n_timed=N_TIMED) -> float:
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
        from tsfast.models.narx import backend_triton

        if backend_triton.is_available():
            names.append("triton")
    else:
        from tsfast.models.narx import backend_c

        if backend_c.is_available():
            names.append("c")
    return names


def make_model(backend, device, args):
    m = NarxMLP(
        N_U,
        N_Y,
        na=args.na,
        nb=args.nb,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        washout=args.washout,
        backend=backend,
    ).to(device)
    return m


def run(args):
    device = torch.device(args.device) if args.device else detect_device()
    torch.manual_seed(SEED)
    names = backends_for(device, args.compiled)
    print(
        f"device={device.type}  na={args.na} nb={args.nb} hidden={args.hidden_size}x{args.num_layers}"
        f"  L={args.seq_len}  (us per trajectory)"
    )
    header = f"{'train step':>14s}" + "".join(f"{'B=' + str(b):>12s}" for b in args.batch_sizes)
    print(header)
    print("-" * len(header))
    for name in names:
        cells = []
        for B in args.batch_sizes:
            m = make_model(name, device, args)
            x = torch.randn(B, args.seq_len, N_U + N_Y, device=device)
            tgt = torch.randn(B, args.seq_len, N_Y, device=device)
            opt = torch.optim.Adam(m.parameters(), lr=1e-3)

            def step():
                opt.zero_grad(set_to_none=True)
                loss = F.mse_loss(m.forward(x, ar=True), tgt)
                loss.backward()
                opt.step()

            cells.append(bench(step, device) / B * 1e3)
        print(f"{name:>14s}" + "".join(f"{c:>12.2f}" for c in cells))

    print(f"\n{'inference':>14s}" + "".join(f"{'B=' + str(b):>12s}" for b in args.batch_sizes))
    print("-" * len(header))
    for name in names:
        cells = []
        for B in args.batch_sizes:
            m = make_model(name, device, args).eval()
            x = torch.randn(B, args.seq_len, N_U + N_Y, device=device)
            with torch.no_grad():
                cells.append(bench(lambda: m.forward(x, ar=True), device) / B * 1e3)
        print(f"{name:>14s}" + "".join(f"{c:>12.2f}" for c in cells))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", default=None, help="cuda or cpu (auto-detected if omitted)")
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[16, 64, 256])
    p.add_argument("--seq-len", type=int, default=300)
    p.add_argument("--na", type=int, default=8)
    p.add_argument("--nb", type=int, default=8)
    p.add_argument("--hidden-size", type=int, default=64)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--washout", type=int, default=50)
    p.add_argument("--compiled", action="store_true", help="also benchmark torch.compile")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
