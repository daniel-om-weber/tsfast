#!/usr/bin/env python
"""Benchmark: R2DN against REN at matched parameter count.

Times a full training step (forward + MSE + backward + Adam) and inference of the
sequential rollout, reported as microseconds per trajectory. Runs on synthetic data —
no dataset download required.

Both models are certified by construction and differ in how they spend nonlinear capacity:
the REN solves an equilibrium layer of ``n_nl`` neurons, which is a sequential sweep even in
the acyclic case, while the R2DN evaluates a ``depth``-layer 1-Lipschitz network, which is a
stack of GEMMs. So the eager REN issues ``seq_len * n_nl`` tiny kernels against the R2DN's
``seq_len * depth``, and the gap is the point of the architecture.

Capacity is matched by parameter count, not by width: for each REN width the script picks the
R2DN width whose model is closest in size, and prints both, along with the total neuron count
each spends its budget on. Both models' fused backends are included where available, and that
is the row pair that matters: eager, the two are separated by dispatch overhead rather than by
their architectures, and the eager rows are here mainly to show how much of the gap that is.

Usage:
    uv run python benchmarks/benchmark_r2dn.py
    uv run python benchmarks/benchmark_r2dn.py --device cpu --n-nl 8 16 32
    uv run python benchmarks/benchmark_r2dn.py --depth 4 --seq-len 1000 --batch-sizes 64
"""

import argparse
import time

import torch
import torch.nn.functional as F

from tsfast.models.architectures.ren import REN, R2DN

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


def n_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


def fused_backend(device: torch.device) -> str | None:
    """The REN's fused backend on this device, if its kernels are usable."""
    if device.type == "cuda":
        from tsfast.models.architectures.ren import backend_triton

        return "triton" if backend_triton.is_available() else None
    from tsfast.models.architectures.ren import backend_c

    return "c" if backend_c.is_available() else None


def r2dn_fused_backend(device: torch.device) -> str | None:
    """The R2DN's fused backend on this device, if its kernels are usable (CUDA only)."""
    if device.type != "cuda":
        return None
    from tsfast.models.architectures.ren import r2dn_backend_triton

    return "triton" if r2dn_backend_triton.is_available() else None


def matched_width(target: int, depth: int) -> int:
    """Smallest R2DN width whose parameter count reaches ``target``."""
    width = 2
    while n_params(R2DN(N_INPUT, N_OUTPUT, n_state=N_STATE, n_nl=width, depth=depth)) < target:
        width += 1
    return width


def make_model(kind: str, backend: str, width: int, depth: int, device: torch.device):
    if kind == "REN":
        return REN(N_INPUT, N_OUTPUT, n_state=N_STATE, n_nl=width, backend=backend).to(device)
    return R2DN(N_INPUT, N_OUTPUT, n_state=N_STATE, n_nl=width, depth=depth, backend=backend).to(device)


def make_train_step(model, batch: int, seq_len: int, device: torch.device):
    u = torch.randn(batch, seq_len, N_INPUT, device=device)
    x0 = torch.zeros(batch, N_STATE, device=device)
    tgt = torch.randn(batch, seq_len, N_OUTPUT, device=device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    def step():
        opt.zero_grad(set_to_none=True)
        F.mse_loss(model(u, x0), tgt).backward()
        opt.step()

    return step


def rows(device: torch.device, include_eager: bool) -> list[tuple[str, str, str]]:
    """``(label, kind, backend)`` for every configuration to time."""
    out = [("REN eager", "REN", "eager"), ("R2DN eager", "R2DN", "eager")] if include_eager else []
    if fused := fused_backend(device):
        out.insert(1 if include_eager else 0, (f"REN {fused}", "REN", fused))
    if fused := r2dn_fused_backend(device):
        out.append((f"R2DN {fused}", "R2DN", fused))
    return out


def run(args):
    device = torch.device(args.device) if args.device else detect_device()
    torch.manual_seed(SEED)
    configs = []
    for width in args.n_nl:
        target = n_params(REN(N_INPUT, N_OUTPUT, n_state=N_STATE, n_nl=width))
        r2dn_width = matched_width(target, args.depth)
        configs.append((width, r2dn_width))

    print(f"device={device.type}  n_state={N_STATE}  L={args.seq_len}  depth={args.depth}  (us per trajectory)\n")
    print(f"{'capacity':>18s}" + "".join(f"{f'nv={a}/{b}':>12s}" for a, b in configs))
    for label, sizes in (
        ("parameters", [n_params(REN(N_INPUT, N_OUTPUT, n_state=N_STATE, n_nl=a)) for a, _ in configs]),
        ("neurons REN/R2DN", [f"{a}/{b * args.depth}" for a, b in configs]),
    ):
        print(f"{label:>18s}" + "".join(f"{s:>12}" for s in sizes))
    print()

    for batch in args.batch_sizes:
        header = f"{'train step B=' + str(batch):>18s}" + "".join(f"{f'nv={a}/{b}':>12s}" for a, b in configs)
        print(header)
        print("-" * len(header))
        for label, kind, backend in rows(device, include_eager=not args.no_eager):
            cells = []
            for ren_width, r2dn_width in configs:
                model = make_model(kind, backend, ren_width if kind == "REN" else r2dn_width, args.depth, device)
                cells.append(bench(make_train_step(model, batch, args.seq_len, device), device) / batch * 1e3)
            print(f"{label:>18s}" + "".join(f"{c:>12.1f}" for c in cells))
        print()

    batch = args.batch_sizes[-1]
    header = f"{'inference B=' + str(batch):>18s}" + "".join(f"{f'nv={a}/{b}':>12s}" for a, b in configs)
    print(header)
    print("-" * len(header))
    for label, kind, backend in rows(device, include_eager=not args.no_eager):
        cells = []
        for ren_width, r2dn_width in configs:
            model = make_model(kind, backend, ren_width if kind == "REN" else r2dn_width, args.depth, device).eval()
            u = torch.randn(batch, args.seq_len, N_INPUT, device=device)
            x0 = torch.zeros(batch, N_STATE, device=device)
            with torch.no_grad():
                cells.append(bench(lambda: model(u, x0), device) / batch * 1e3)
        print(f"{label:>18s}" + "".join(f"{c:>12.1f}" for c in cells))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", default=None, help="cuda or cpu (auto-detected if omitted)")
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[16, 64])
    p.add_argument("--seq-len", type=int, default=300)
    p.add_argument("--depth", type=int, default=2, help="nonlinear layers in the R2DN's network")
    p.add_argument("--n-nl", type=int, nargs="+", default=[8, 16, 32, 64], help="REN widths to scan")
    p.add_argument("--no-eager", action="store_true", help="skip the eager rows (the REN's is slow at large nv)")
    run(p.parse_args())


if __name__ == "__main__":
    main()
