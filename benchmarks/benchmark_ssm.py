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
N_OUTPUT = 10
N_WARMUP = 10
N_TIMED = 30
SEED = 42


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
    elif device.type == "mps":
        from tsfast.models.ssm import backend_metal as ssm_metal

        if ssm_metal.is_available():
            names.append("metal")
    else:
        from tsfast.models.ssm import backend_c as ssm_c

        if ssm_c.is_available():
            names.append("c")
    return names


def make_train_step(name, device, hidden, B, seq_len):
    """Build a training-step closure for a backend name ('<backend>' or '<backend>+graph')."""
    backend, _, graphed = name.partition("+")
    m = NeuralStateSpace(N_INPUT, N_OUTPUT, N_STATE, hidden, backend=backend, return_state=bool(graphed)).to(device)
    model = m
    if graphed:
        from tsfast.models.cudagraph import GraphedStatefulModel

        model = GraphedStatefulModel(m)
    u = torch.randn(B, seq_len, N_INPUT, device=device)
    x0 = torch.zeros(B, N_STATE, device=device)
    tgt = torch.randn(B, seq_len, N_OUTPUT, device=device)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)

    def step():
        opt.zero_grad(set_to_none=True)
        pred = model(u) if graphed else model(u, x0)
        if graphed:
            pred = pred[0]
        loss = F.mse_loss(pred, tgt)
        loss.backward()
        opt.step()

    return step


def run(args):
    device = torch.device(args.device) if args.device else detect_device()
    torch.manual_seed(SEED)
    hidden = list(args.hidden)
    names = backends_for(device, args.compiled)
    if args.graphed and device.type == "cuda":
        names += [f"{n}+graph" for n in names if n != "compiled"]
    print(f"device={device.type}  hidden={hidden}  L={args.seq_len}  (us per trajectory)")
    header = f"{'train step':>14s}" + "".join(f"{'B=' + str(b):>12s}" for b in args.batch_sizes)
    print(header)
    print("-" * len(header))
    for name in names:
        cells = []
        for B in args.batch_sizes:
            step = make_train_step(name, device, hidden, B, args.seq_len)
            cells.append(bench(step, device) / B * 1e3)
        print(f"{name:>14s}" + "".join(f"{c:>12.2f}" for c in cells))

    print(f"\n{'inference':>14s}" + "".join(f"{'B=' + str(b):>12s}" for b in args.batch_sizes))
    print("-" * len(header))
    for name in [n for n in names if "+" not in n]:  # graph capture only pays off in training
        cells = []
        for B in args.batch_sizes:
            m = NeuralStateSpace(N_INPUT, N_OUTPUT, N_STATE, hidden, backend=name).to(device).eval()
            u = torch.randn(B, args.seq_len, N_INPUT, device=device)
            x0 = torch.zeros(B, N_STATE, device=device)
            with torch.no_grad():
                cells.append(bench(lambda: m(u, x0), device) / B * 1e3)
        print(f"{name:>14s}" + "".join(f"{c:>12.2f}" for c in cells))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", default=None, help="cuda or cpu (auto-detected if omitted)")
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[16, 64, 256])
    p.add_argument("--seq-len", type=int, default=300)
    p.add_argument("--hidden", type=int, nargs="+", default=[64, 64])
    p.add_argument("--compiled", action="store_true", help="include the torch.compile backend (slow first call)")
    p.add_argument("--graphed", action="store_true", help="also wrap CUDA backends in GraphedStatefulModel")
    run(p.parse_args())


if __name__ == "__main__":
    main()
