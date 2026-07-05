#!/usr/bin/env python
"""Benchmark: constant-coefficient diagonal scan kernels (triton/c) vs the doubling scan.

Times a fused forward + backward of ``scan.diagonal_recurrence`` — the compute core of the
LRU and S5 families — on model-realistic complex64 shapes, and reports wall-clock per call and
peak memory alongside the speedup over the pure-PyTorch doubling scan. The kernel backends
stream the sequence once where the doubling scan re-reads it ~log2(L) times.

Usage:
    uv run python benchmarks/benchmark_diagonal_scan.py
    uv run python benchmarks/benchmark_diagonal_scan.py --device cpu --seq-len 8000
"""

import argparse
import time

import torch

import tsfast.models._core.scan as scan

N_WARMUP = 10
N_TIMED = 30
SEED = 42


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def bench(fn, device):
    for _ in range(N_WARMUP):
        fn()
    sync(device)
    t0 = time.perf_counter()
    for _ in range(N_TIMED):
        fn()
    sync(device)
    return (time.perf_counter() - t0) / N_TIMED * 1e3


def peak_mem_mb(fn, device):
    if device.type != "cuda":
        return float("nan")
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1024**2


def make_step(backend, lam, v, x0):
    def step():
        lam_ = lam.detach().clone().requires_grad_()
        v_ = v.detach().clone().requires_grad_()
        scan.backend = backend
        out = scan.diagonal_recurrence(lam_, v_, x0)
        (out.abs() ** 2).sum().backward()

    return step


def run(args):
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED)
    kernel = "triton" if device.type == "cuda" else "c"
    B, L = args.batch, args.seq_len
    print(f"device={device.type}  kernel={kernel}  B={B}  L={L}  (fwd+bwd, complex64)")
    header = f"{'n_state':>8s}{'doubling ms':>14s}{kernel + ' ms':>14s}{'speedup':>10s}"
    if device.type == "cuda":
        header += f"{'dbl MB':>10s}{kernel + ' MB':>10s}"
    print(header)
    print("-" * len(header))
    for n in args.states:
        lam = (torch.rand(n) * 0.9 * torch.exp(1j * torch.rand(n) * 3.0)).to(torch.complex64).to(device)
        v = torch.randn(B, L, n, dtype=torch.complex64, device=device)
        x0 = torch.randn(B, n, dtype=torch.complex64, device=device)
        t_dbl = bench(make_step("doubling", lam, v, x0), device)
        t_k = bench(make_step(kernel, lam, v, x0), device)
        row = f"{n:>8d}{t_dbl:>14.3f}{t_k:>14.3f}{t_dbl / t_k:>9.1f}x"
        if device.type == "cuda":
            m_dbl = peak_mem_mb(make_step("doubling", lam, v, x0), device)
            m_k = peak_mem_mb(make_step(kernel, lam, v, x0), device)
            row += f"{m_dbl:>10.0f}{m_k:>10.0f}"
        print(row)
    scan.backend = "auto"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", default=None, help="cuda or cpu (auto-detected if omitted)")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--seq-len", type=int, default=1600)
    p.add_argument("--states", type=int, nargs="+", default=[32, 64, 128])
    run(p.parse_args())


if __name__ == "__main__":
    main()
