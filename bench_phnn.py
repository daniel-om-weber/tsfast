"""Benchmark the PHNN rollout backends on the profiled WH config.

Profiled training step: n_state 8, hidden 64, num_layers 2, section_len 120, bs 32.
Reports ms/step for forward+backward (train) on the requested backends.

Usage: python bench_phnn.py [device] [backend ...]
  device: cuda | cpu   (default cuda)
"""

import sys
import time

import torch

from tsfast.models.phnn import PHNN


def make_model(backend):
    m = PHNN(1, 1, n_state=8, hidden_size=64, num_layers=2, dt=0.1, n_init=50, backend=backend)
    for p in m.parameters():
        torch.nn.init.normal_(p, std=0.1)
    return m


def bench(backend, device, L=120, bs=32, iters=30, warmup=5, train=True):
    torch.manual_seed(0)
    m = make_model(backend).to(device)
    seq = 50 + L
    x = torch.randn(bs, seq, 2, device=device)
    cuda = device == "cuda"

    def step():
        out = m(x)
        if train:
            loss = out[:, 50:].pow(2).mean()
            m.zero_grad(set_to_none=True)
            loss.backward()

    for _ in range(warmup):
        step()
    if cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        step()
    if cuda:
        torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters * 1e3
    return dt


if __name__ == "__main__":
    device = sys.argv[1] if len(sys.argv) > 1 else "cuda"
    backends = sys.argv[2:] or (["compiled", "triton"] if device == "cuda" else ["eager", "c"])
    mode = "fwd+bwd"
    print(f"device={device}  config: n_state8 hidden64 layers2 L120 bs32  ({mode})")
    for b in backends:
        try:
            ms = bench(b, device)
            print(f"  {b:10s} {ms:8.2f} ms/step")
        except Exception as e:  # noqa: BLE001
            print(f"  {b:10s} FAILED: {type(e).__name__}: {e}")
