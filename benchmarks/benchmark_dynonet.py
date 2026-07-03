#!/usr/bin/env python
"""Benchmark: DynoNet G-block execution backends (eager loop vs log-doubling scan).

Times a full training step (forward + MSE + backward + Adam) and inference of the
block-oriented model, reported as microseconds per trajectory. The scan's advantage
grows with sequence length, so try --seq-len 1000 and beyond. Runs on synthetic
data — no dataset download required.

Usage:
    uv run python benchmarks/benchmark_dynonet.py
    uv run python benchmarks/benchmark_dynonet.py --device cpu --batch-sizes 8 32
    uv run python benchmarks/benchmark_dynonet.py --seq-len 4096 --na 3 --nb 8
"""

import argparse
import time

import torch
import torch.nn.functional as F

from tsfast.models.dynonet import DynoNet

N_INPUT = 1
N_OUTPUT = 1
N_WARMUP = 10
N_TIMED = 30
SEED = 42

BACKENDS = ["eager", "scan"]


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


def make_model(args, backend, device) -> DynoNet:
    m = DynoNet(N_INPUT, N_OUTPUT, n_channels=args.channels, nb=args.nb, na=args.na, backend=backend).to(device)
    with torch.no_grad():
        for mod in m.modules():
            if hasattr(mod, "a_coeff"):
                mod.a_coeff.uniform_(-0.3, 0.3)  # exercise the IIR path with non-trivial poles
    return m


def make_train_step(args, backend, device, B):
    m = make_model(args, backend, device)
    u = torch.randn(B, args.seq_len, N_INPUT, device=device)
    tgt = torch.randn(B, args.seq_len, N_OUTPUT, device=device)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)

    def step():
        opt.zero_grad(set_to_none=True)
        loss = F.mse_loss(m(u), tgt)
        loss.backward()
        opt.step()

    return step


def run(args):
    device = torch.device(args.device) if args.device else detect_device()
    torch.manual_seed(SEED)
    print(
        f"device={device.type}  channels={args.channels}  nb={args.nb}  na={args.na}  "
        f"L={args.seq_len}  (us per trajectory)"
    )
    header = f"{'train step':>14s}" + "".join(f"{'B=' + str(b):>12s}" for b in args.batch_sizes)
    print(header)
    print("-" * len(header))
    for backend in BACKENDS:
        cells = []
        for B in args.batch_sizes:
            step = make_train_step(args, backend, device, B)
            cells.append(bench(step, device) / B * 1e3)
        print(f"{backend:>14s}" + "".join(f"{c:>12.2f}" for c in cells))

    print(f"\n{'inference':>14s}" + "".join(f"{'B=' + str(b):>12s}" for b in args.batch_sizes))
    print("-" * len(header))
    for backend in BACKENDS:
        cells = []
        for B in args.batch_sizes:
            m = make_model(args, backend, device).eval()
            u = torch.randn(B, args.seq_len, N_INPUT, device=device)
            with torch.no_grad():
                cells.append(bench(lambda: m(u), device) / B * 1e3)
        print(f"{backend:>14s}" + "".join(f"{c:>12.2f}" for c in cells))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", default=None, help="cuda or cpu (auto-detected if omitted)")
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[16, 64, 256])
    p.add_argument("--seq-len", type=int, default=1000)
    p.add_argument("--channels", type=int, default=8)
    p.add_argument("--nb", type=int, default=8)
    p.add_argument("--na", type=int, default=2)
    run(p.parse_args())


if __name__ == "__main__":
    main()
