"""Forward+backward benchmark: fused selective-scan kernels vs the doubling scan.

Reports median wall time and peak memory for the raw ``selective_recurrence`` op. GPU cases
compare the Triton kernel against the doubling scan; the CPU case compares the C++ kernel.
Run as: PYTHONPATH=<wt> CUDA_VISIBLE_DEVICES=0 uv run python benchmarks/bench_selective.py
"""

import resource
import statistics
import time

import torch

import tsfast.models._core.scan as scan


def _bench(shape, device, iters=8, warmup=3):
    B, L, N = shape
    torch.manual_seed(0)
    lam0 = (torch.rand(B, L, N, device=device) * 0.9 + 0.05).requires_grad_()
    v0 = torch.randn(B, L, N, device=device).requires_grad_()
    g = torch.randn(B, L, N, device=device)
    cuda = device == "cuda"

    def once():
        lam0.grad = v0.grad = None
        out = scan.selective_recurrence(lam0, v0)
        out.backward(g)

    for _ in range(warmup):
        once()
    if cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    else:
        rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    times = []
    for _ in range(iters):
        if cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        once()
        if cuda:
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e3)

    if cuda:
        peak = torch.cuda.max_memory_allocated() / 1e6
    else:
        peak = (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - rss0) / 1e3
    return statistics.median(times), peak


def main():
    rows = []
    if torch.cuda.is_available():
        for shape in [(32, 1600, 1024), (128, 1600, 1024)]:
            scan.backend = "doubling"
            t_d, m_d = _bench(shape, "cuda")
            scan.backend = "auto"
            t_k, m_k = _bench(shape, "cuda")
            rows.append(("triton", shape, t_d, m_d, t_k, m_k))
    else:
        print("CUDA unavailable; skipping GPU cases")

    for shape in [(8, 6841, 512)]:
        scan.backend = "doubling"
        t_d, m_d = _bench(shape, "cpu")
        scan.backend = "auto"
        t_k, m_k = _bench(shape, "cpu")
        rows.append(("c", shape, t_d, m_d, t_k, m_k))

    scan.backend = "auto"
    hdr = f"{'kernel':7s} {'shape':18s} {'doubling ms':>12s} {'dbl peakMB':>11s} " f"{'kernel ms':>10s} {'ker peakMB':>11s} {'speedup':>8s}"
    print(hdr)
    print("-" * len(hdr))
    for kern, shape, t_d, m_d, t_k, m_k in rows:
        print(
            f"{kern:7s} {str(shape):18s} {t_d:12.2f} {m_d:11.1f} {t_k:10.2f} {m_k:11.1f} {t_d / t_k:7.1f}x"
        )


if __name__ == "__main__":
    main()
