#!/usr/bin/env python
"""Benchmark: REN execution backends (eager vs c vs triton).

Times a full training step (forward + MSE + backward + Adam) and inference of the
sequential rollout, reported as microseconds per trajectory. Runs on synthetic data —
no dataset download required.

The REN rollout is sequential twice over — along the sequence and along the ``n_nl``
neurons of the equilibrium layer — so the eager path issues ``seq_len * n_nl`` tiny
kernels and is bounded entirely by per-op dispatch. The sweep width is therefore the
axis to scan: ``--n-nl`` shows where the fused kernels pull away.

Usage:
    uv run python benchmarks/benchmark_ren.py
    uv run python benchmarks/benchmark_ren.py --device cpu --n-nl 8 16 32
    uv run python benchmarks/benchmark_ren.py --seq-len 1000 --batch-sizes 64
"""

import argparse
import time

import torch
import torch.nn.functional as F

from tsfast.models.architectures.ren import REN

N_INPUT = 2
N_OUTPUT = 2
N_STATE = 8
N_WARMUP = 3
N_TIMED = 10
SEED = 42


def detect_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def backends_for(device: torch.device, include_eager: bool) -> list[str]:
    names = ["eager"] if include_eager else []
    if device.type == "cuda":
        from tsfast.models.architectures.ren import backend_triton

        if backend_triton.is_available():
            names.append("triton")
    else:
        from tsfast.models.architectures.ren import backend_c

        if backend_c.is_available():
            names.append("c")
    return names


def make_model(backend, device, n_nl):
    return REN(N_INPUT, N_OUTPUT, n_state=N_STATE, n_nl=n_nl, backend=backend).to(device)


def make_train_step(backend, device, n_nl, batch, seq_len):
    m = make_model(backend, device, n_nl)
    u = torch.randn(batch, seq_len, N_INPUT, device=device)
    x0 = torch.zeros(batch, N_STATE, device=device)
    tgt = torch.randn(batch, seq_len, N_OUTPUT, device=device)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)

    def step():
        opt.zero_grad(set_to_none=True)
        F.mse_loss(m(u, x0), tgt).backward()
        opt.step()

    return step


def run(args):
    device = torch.device(args.device) if args.device else detect_device()
    torch.manual_seed(SEED)
    names = backends_for(device, include_eager=not args.no_eager)
    print(f"device={device.type}  n_state={N_STATE}  L={args.seq_len}  (us per trajectory)\n")
    for batch in args.batch_sizes:
        header = f"{'train step B=' + str(batch):>18s}" + "".join(f"{'nv=' + str(v):>12s}" for v in args.n_nl)
        print(header)
        print("-" * len(header))
        for name in names:
            cells = []
            for n_nl in args.n_nl:
                step = make_train_step(name, device, n_nl, batch, args.seq_len)
                cells.append(bench(step, device) / batch * 1e3)
            print(f"{name:>18s}" + "".join(f"{c:>12.1f}" for c in cells))
        print()

    batch = args.batch_sizes[-1]
    header = f"{'inference B=' + str(batch):>18s}" + "".join(f"{'nv=' + str(v):>12s}" for v in args.n_nl)
    print(header)
    print("-" * len(header))
    for name in names:
        cells = []
        for n_nl in args.n_nl:
            m = make_model(name, device, n_nl).eval()
            u = torch.randn(batch, args.seq_len, N_INPUT, device=device)
            x0 = torch.zeros(batch, N_STATE, device=device)
            with torch.no_grad():
                cells.append(bench(lambda: m(u, x0), device) / batch * 1e3)
        print(f"{name:>18s}" + "".join(f"{c:>12.1f}" for c in cells))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", default=None, help="cuda or cpu (auto-detected if omitted)")
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[16, 64])
    p.add_argument("--seq-len", type=int, default=300)
    p.add_argument("--n-nl", type=int, nargs="+", default=[8, 16, 32, 64], help="equilibrium-layer widths to scan")
    p.add_argument("--no-eager", action="store_true", help="skip the eager baseline (it is slow at large nv)")
    run(p.parse_args())


if __name__ == "__main__":
    main()
