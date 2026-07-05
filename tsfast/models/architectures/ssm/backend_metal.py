"""Generated-Metal execution backend for NeuralStateSpace: persistent MPS rollout and BPTT.

The rollout runs as ONE Metal kernel launch: all L steps in a single compute dispatch, one
threadgroup per trajectory with one thread per neuron of the widest layer. Each thread keeps
its own weight row in registers for the whole rollout, activations pass between layers
through threadgroup memory, and the next step's input is prefetched a full step ahead so the
device-memory load latency never sits in the sequential dependency chain. Kernels are
generated from the layer spec (dims baked as constants) and compiled at runtime via
``torch.mps.compile_shader``.

Backward follows the split-BPTT design shared with the C and Triton backends: a reverse-sweep
kernel carries the state adjoint (each thread holding the weight column it needs in
registers) and emits the per-step pre-activation adjoints; the parameter gradients are
batched MPS GEMMs over the ``[B*L, .]`` flattened adjoints (``mlp_param_grads``).

Unlike the nonlinear forward, the state-adjoint recurrence is LINEAR in the adjoint
(``gy_t = gout_t + J^T_{t+1} gy_{t+1}`` with per-step Jacobians of the transition MLP), so
when the batch alone cannot fill the GPU the backward is additionally parallelized over the
sequence: the Jacobians are built for all steps at once by batched GEMMs, parallel chunks
reduce to boundary summaries (matrix suffix products), a short scan propagates the adjoint
across chunk boundaries, and the reverse sweep then runs in every chunk concurrently. This
cuts the sequential depth from L to roughly L/chunks and is exact — no approximation.
"""

__all__ = [
    "metal_rollout",
    "is_available",
    "fits",
]

import torch

from .core import SSMSpec, check_rollout_args, mlp_param_grads

_ACT_MSL = {
    "tanh": ("precise::tanh({a})", "(1.0f - {z} * {z})"),
    "sigmoid": ("1.0f / (1.0f + precise::exp(-({a})))", "({z} * (1.0f - {z}))"),
    "relu": ("fmax({a}, 0.0f)", "({z} > 0.0f ? 1.0f : 0.0f)"),
}

_DACT_TORCH = {
    "tanh": lambda z: 1.0 - z * z,
    "sigmoid": lambda z: z * (1.0 - z),
    "relu": lambda z: (z > 0).to(z.dtype),
}

# One thread per neuron with its weight row/column in registers: beyond this width the
# register pressure defeats the design (and Metal's buffer-binding count caps the depth).
_MAX_WIDTH = 128
_MAX_LINEAR = 8

# Sequence-parallel backward: engaged when the batch alone leaves the GPU latency-bound
# (measured crossover: wins up to B=16, loses from B=64 where the plain sweep is already
# throughput-bound and the Jacobian build is pure extra work). The state must be small
# enough for NX x NX chunk products in threadgroup memory, and the materialized Jacobians
# [B, L, NX, NX] must stay a reasonable fraction of memory.
_SCAN_MAX_STATE = 32
_SCAN_MAX_BATCH = 16
_SCAN_MIN_CHUNK = 64
_SCAN_TARGET_GROUPS = 512
_SCAN_MAX_JT_BYTES = 256 * 1024**2

_LIBS: dict[SSMSpec, object] = {}
_AVAILABLE: bool | None = None


def is_available() -> bool:
    """True if MPS is present and runtime Metal shader compilation works."""
    global _AVAILABLE
    if _AVAILABLE is None:
        _AVAILABLE = _probe()
    return _AVAILABLE


def _probe() -> bool:
    if not (torch.backends.mps.is_available() and hasattr(torch.mps, "compile_shader")):
        return False
    try:
        torch.mps.compile_shader("kernel void probe(device float* o, uint i [[thread_position_in_grid]]) {}")
    except Exception:
        return False
    return True


def fits(spec: SSMSpec) -> bool:
    """Whether the register-resident kernels apply: every layer width must stay on-chip."""
    return spec.act in _ACT_MSL and max(spec.dims) <= _MAX_WIDTH and spec.n_linear <= _MAX_LINEAR


def _tpg(spec: SSMSpec) -> int:
    """Threads per threadgroup: the widest layer output, rounded up to full simdgroups."""
    return max(32, -(-max(spec.dims[1:]) // 32) * 32)


def _gen_source(spec: SSMSpec) -> str:
    """Emit the spec-specialized Metal kernels (inference fwd, training fwd, bwd)."""
    dims = spec.dims
    nx, nu, k = spec.n_state, spec.n_input, spec.n_linear
    tpg = _tpg(spec)
    act, dact = _ACT_MSL[spec.act]
    lines = [
        "#include <metal_stdlib>",
        "using namespace metal;",
        f"constant uint NX = {nx};",
        f"constant uint NU = {nu};",
        "",
    ]

    def fwd_kernel(name: str, store_z: bool) -> list[str]:
        w_args = "".join(f"    device const float* w{i}, device const float* c{i},\n" for i in range(k))
        z_args = "".join(f"    device float* z{i},\n" for i in range(k - 1)) if store_z else ""
        out = [
            f"kernel void {name}(",
            "    device const float* u, device const float* x0,",
            w_args + "    device float* out,",
            z_args + "    constant uint& L,",
            "    uint tid [[thread_position_in_threadgroup]],",
            "    uint b   [[threadgroup_position_in_grid]])",
            "{",
            f"    threadgroup float xu[{nx + nu}];",
        ]
        for i in range(k - 1):
            out.append(f"    threadgroup float h{i}[{dims[i + 1]}];")
        if k == 1:
            out.append(f"    threadgroup float xn[{nx}];")
        for i in range(k):
            out += [
                f"    float w{i}r[{dims[i]}];",
                f"    float c{i}r = 0.0f;",
                f"    if (tid < {dims[i + 1]}) {{",
                f"        for (uint j = 0; j < {dims[i]}; ++j) w{i}r[j] = w{i}[tid*{dims[i]} + j];",
                f"        c{i}r = c{i}[tid];",
                "    }",
            ]
        out.append("    if (tid < NX) xu[tid] = x0[b*NX + tid];")
        # u(t+1) is loaded into a register one full step ahead of its threadgroup staging, so
        # the device-memory latency overlaps a whole step of compute.
        prefetch = nu <= tpg
        if prefetch:
            out += [
                "    if (tid < NU) xu[NX + tid] = u[b*L*NU + tid];",
                "    float ur = (tid < NU) ? u[(b*L + min(1u, L - 1))*NU + tid] : 0.0f;",
            ]
        else:
            out.append(f"    for (uint j = tid; j < NU; j += {tpg}) xu[NX + j] = u[b*L*NU + j];")
        out += [
            "    threadgroup_barrier(mem_flags::mem_threadgroup);",
            "",
            "    for (uint t = 0; t < L; ++t) {",
        ]

        def emit_stage_u(dst: list[str]):
            if prefetch:
                dst += [
                    "        if (tid < NU) xu[NX + tid] = ur;",
                    "        ur = (tid < NU) ? u[(b*L + min(t + 2, L - 1))*NU + tid] : 0.0f;",
                ]
            else:
                dst.append(
                    f"        for (uint j = tid; j < NU; j += {tpg}) xu[NX + j] = u[(b*L + min(t + 1, L - 1))*NU + j];"
                )

        for i in range(k):
            src = "xu" if i == 0 else f"h{i - 1}"
            last = i == k - 1
            dst = ("xn" if k == 1 else "xu") if last else f"h{i}"
            out.append(f"        if (tid < {dims[i + 1]}) {{")
            out.append(f"            float acc = c{i}r;")
            out.append(f"            for (uint j = 0; j < {dims[i]}; ++j) acc += w{i}r[j] * {src}[j];")
            if last:
                out.append(f"            {dst}[tid] = acc;")
                out.append("            out[(b*L + t)*NX + tid] = acc;")
            else:
                out.append("            acc = " + act.format(a="acc") + ";")
                out.append(f"            {dst}[tid] = acc;")
                if store_z:
                    out.append(f"            z{i}[(b*L + t)*{dims[i + 1]} + tid] = acc;")
            out.append("        }")
            if i == 0 and k >= 2:
                # staged here so the write to xu's input slots lands after layer 0 consumed
                # them (the barrier below) and before the next step reads them
                out.append("        threadgroup_barrier(mem_flags::mem_threadgroup);")
                emit_stage_u(out)
            elif not last:
                out.append("        threadgroup_barrier(mem_flags::mem_threadgroup);")
        out.append("        threadgroup_barrier(mem_flags::mem_threadgroup);")
        if k == 1:
            out.append("        if (tid < NX) xu[tid] = xn[tid];")
            emit_stage_u(out)
            out.append("        threadgroup_barrier(mem_flags::mem_threadgroup);")
        out += ["    }", "}", ""]
        return out

    def bwd_kernel() -> list[str]:
        z_args = "".join(f"    device const float* z{i},\n" for i in range(k - 1))
        w_args = "".join(f"    device const float* w{i},\n" for i in range(k))
        ga_args = "".join(f"    device float* ga{i},\n" for i in range(k - 1))
        out = [
            "kernel void ssm_bwd(",
            "    device const float* gout,",
            z_args + w_args + "    device const float* jt, device const float* bnd,",
            "    device float* gy,",
            ga_args + "    device float* gx0,",
            "    constant uint& L, constant uint& CL, constant uint& C,",
            "    uint tid [[thread_position_in_threadgroup]],",
            "    uint g   [[threadgroup_position_in_grid]])",
            "{",
            "    const uint b = g / C, c = g % C;",
            "    const uint t0 = c * CL;",
            "    const uint t1 = min(t0 + CL, L);",
            f"    threadgroup float gys[{nx}];",
        ]
        for i in range(k - 1):
            out.append(f"    threadgroup float gh{i}[{dims[i + 1]}];")
        # weight COLUMNS this time: backward multiplies by the transpose
        for li in range(k):
            width = nx if li == 0 else dims[li]
            out += [
                f"    float wt{li}r[{dims[li + 1]}];",
                f"    if (tid < {width}) {{",
                f"        for (uint o = 0; o < {dims[li + 1]}; ++o) wt{li}r[o] = w{li}[o*{dims[li]} + tid];",
                "    }",
            ]
        out += [
            "    float carry = 0.0f;",
            "    if (tid < NX && t1 < L) {",
            "        // carry entering step t1-1 is J^T_{t1} times the boundary adjoint gy_{t1}",
            "        for (uint m = 0; m < NX; ++m)",
            "            carry += jt[((b*L + t1)*NX + tid)*NX + m] * bnd[(b*(C + 1) + c + 1)*NX + m];",
            "    }",
            "    float goutr = (tid < NX) ? gout[(b*L + t1 - 1)*NX + tid] : 0.0f;",
        ]
        for i in range(k - 1):
            out.append(f"    float z{i}r = (tid < {dims[i + 1]}) ? z{i}[(b*L + t1 - 1)*{dims[i + 1]} + tid] : 0.0f;")
        out += [
            "",
            "    for (uint ti = t1; ti > t0; --ti) {",
            "        const uint t = ti - 1;",
            "        if (tid < NX) {",
            "            const float gyv = goutr + carry;",
            "            gys[tid] = gyv;",
            "            gy[(b*L + t)*NX + tid] = gyv;",
            "        }",
            "        goutr = (tid < NX) ? gout[(b*L + (t > 0 ? t - 1 : 0))*NX + tid] : 0.0f;",
            "        threadgroup_barrier(mem_flags::mem_threadgroup);",
        ]
        prev = "gys"
        for li in range(k - 1, 0, -1):
            i = li - 1  # hidden index whose adjoint this emits
            out += [
                f"        if (tid < {dims[li]}) {{",
                "            float acc = 0.0f;",
                f"            for (uint o = 0; o < {dims[li + 1]}; ++o) acc += wt{li}r[o] * {prev}[o];",
                "            const float g = acc * " + dact.format(z=f"z{i}r") + ";",
                f"            gh{i}[tid] = g;",
                f"            ga{i}[(b*L + t)*{dims[li]} + tid] = g;",
                "        }",
                f"        z{i}r = (tid < {dims[li]}) ? z{i}[(b*L + (t > 0 ? t - 1 : 0))*{dims[li]} + tid] : 0.0f;",
                "        threadgroup_barrier(mem_flags::mem_threadgroup);",
            ]
            prev = f"gh{i}"
        out += [
            "        if (tid < NX) {",
            "            float acc = 0.0f;",
            f"            for (uint o = 0; o < {dims[1]}; ++o) acc += wt0r[o] * {prev}[o];",
            "            carry = acc;",
            "        }",
        ]
        if k == 1:
            # the carry stage reads gys, which the next iteration overwrites
            out.append("        threadgroup_barrier(mem_flags::mem_threadgroup);")
        out += [
            "    }",
            "    if (tid < NX && t0 == 0) gx0[b*NX + tid] = carry;",
            "}",
        ]
        return out

    def scan_kernels() -> str:
        nn = nx * nx
        return f"""
kernel void ssm_bwd_summary(
    device const float* jt, device const float* gout,
    device float* pc, device float* sc,
    constant uint& L, constant uint& CL, constant uint& C,
    uint tid [[thread_position_in_threadgroup]],
    uint g   [[threadgroup_position_in_grid]])
{{
    const uint b = g / C, c = g % C;
    const uint t0 = c * CL;
    const uint t1 = min(t0 + CL, L);
    threadgroup float P[2][{nn}];
    threadgroup float s[2][{nx}];
    threadgroup float A[{nn}];
    for (uint e = tid; e < {nn}; e += 32) P[0][e] = (e / NX == e % NX) ? 1.0f : 0.0f;
    if (tid < NX) s[0][tid] = 0.0f;
    uint cur = 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint ti = t1; ti > t0; --ti) {{
        const uint t = ti - 1;
        if (t + 1 < L) {{
            for (uint e = tid; e < {nn}; e += 32) A[e] = jt[(b*L + t + 1)*{nn} + e];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint e = tid; e < {nn}; e += 32) {{
                const uint i = e / NX, j = e % NX;
                float acc = 0.0f;
                for (uint m = 0; m < NX; ++m) acc += A[i*NX + m] * P[cur][m*NX + j];
                P[1 - cur][e] = acc;
            }}
            if (tid < NX) {{
                float acc = gout[(b*L + t)*NX + tid];
                for (uint m = 0; m < NX; ++m) acc += A[tid*NX + m] * s[cur][m];
                s[1 - cur][tid] = acc;
            }}
        }} else {{
            // the recurrence terminates at t = L-1: gy = gout, no carry from beyond
            for (uint e = tid; e < {nn}; e += 32) P[1 - cur][e] = 0.0f;
            if (tid < NX) s[1 - cur][tid] = gout[(b*L + t)*NX + tid];
        }}
        cur = 1 - cur;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}
    for (uint e = tid; e < {nn}; e += 32) pc[(b*C + c)*{nn} + e] = P[cur][e];
    if (tid < NX) sc[(b*C + c)*NX + tid] = s[cur][tid];
}}

kernel void ssm_bwd_scan(
    device const float* pc, device const float* sc,
    device float* bnd,
    constant uint& C,
    uint tid [[thread_position_in_threadgroup]],
    uint b   [[threadgroup_position_in_grid]])
{{
    threadgroup float v[2][{nx}];
    if (tid < NX) {{
        v[0][tid] = 0.0f;
        bnd[(b*(C + 1) + C)*NX + tid] = 0.0f;
    }}
    uint cur = 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint ci = C; ci > 0; --ci) {{
        const uint c = ci - 1;
        if (tid < NX) {{
            float acc = sc[(b*C + c)*NX + tid];
            for (uint m = 0; m < NX; ++m) acc += pc[(b*C + c)*{nn} + tid*NX + m] * v[cur][m];
            v[1 - cur][tid] = acc;
            bnd[(b*(C + 1) + c)*NX + tid] = acc;
        }}
        cur = 1 - cur;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}
}}
"""

    lines += fwd_kernel("ssm_fwd", store_z=False)
    lines += fwd_kernel("ssm_fwd_train", store_z=True)
    lines += bwd_kernel()
    if nx <= _SCAN_MAX_STATE:
        lines.append(scan_kernels())
    return "\n".join(lines)


def _scan_chunks(spec: SSMSpec, B: int, L: int) -> int:
    """Number of sequence chunks for the backward; 1 disables the scan path."""
    nx = spec.n_state
    if spec.n_linear < 2 or nx > _SCAN_MAX_STATE or B > _SCAN_MAX_BATCH:
        return 1
    if B * L * nx * nx * 4 > _SCAN_MAX_JT_BYTES:
        return 1
    return max(1, min(-(-_SCAN_TARGET_GROUPS // B), L // _SCAN_MIN_CHUNK))


@torch.no_grad()
def _build_jt(spec: SSMSpec, weights: list[torch.Tensor], zs: list[torch.Tensor]) -> torch.Tensor:
    """Per-step transposed state Jacobians ``J^T_t = W0x^T D0 W1^T D1 ... W_{k-1}^T`` as [B, L, NX, NX].

    The diagonal activation-derivative factors come from the stored hidden activations, so
    all L Jacobians are produced by k-1 flat GEMMs — fully parallel over the sequence.
    """
    nx, k = spec.n_state, spec.n_linear
    B, L = zs[0].shape[0], zs[0].shape[1]
    dact = _DACT_TORCH[spec.act]
    n = weights[0][:, :nx].t().unsqueeze(0) * dact(zs[0]).reshape(B * L, 1, -1)
    for i in range(1, k):
        n = (n.reshape(B * L * nx, -1) @ weights[i].t()).reshape(B * L, nx, -1)
        if i < k - 1:
            n = n * dact(zs[i]).reshape(B * L, 1, -1)
    return n.reshape(B, L, nx, nx).contiguous()


def _get_lib(spec: SSMSpec):
    lib = _LIBS.get(spec)
    if lib is None:
        lib = torch.mps.compile_shader(_gen_source(spec))
        _LIBS[spec] = lib
    return lib


def _run_fwd(lib, spec: SSMSpec, u, x0, params, store_z):
    B, L = u.shape[0], u.shape[1]
    tpg = _tpg(spec)
    out = torch.empty(B, L, spec.n_state, device=u.device, dtype=torch.float32)
    wb = [t.detach().contiguous() for t in params]
    if store_z:
        zs = [torch.empty(B, L, h, device=u.device, dtype=torch.float32) for h in spec.hidden]
        lib.ssm_fwd_train(u, x0, *wb, out, *zs, L, threads=B * tpg, group_size=tpg)
    else:
        zs = []
        lib.ssm_fwd(u, x0, *wb, out, L, threads=B * tpg, group_size=tpg)
    return out, zs


class _MetalSSMRollout(torch.autograd.Function):
    @staticmethod
    def forward(ctx, lib, spec, u, x0, *params):
        u = u.contiguous()
        x0 = x0.contiguous()
        out, zs = _run_fwd(lib, spec, u, x0, params, store_z=True)
        ctx.lib, ctx.spec = lib, spec
        ctx.save_for_backward(u, x0, out, *zs, *params[0::2])
        return out

    @staticmethod
    def backward(ctx, grad_out):
        lib, spec = ctx.lib, ctx.spec
        k = spec.n_linear
        saved = ctx.saved_tensors
        u, x0, out = saved[0], saved[1], saved[2]
        zs = list(saved[3 : 3 + k - 1])
        weights = list(saved[3 + k - 1 :])
        B, L = u.shape[0], u.shape[1]
        nx = spec.n_state
        tpg = _tpg(spec)
        ws = [w.detach().contiguous() for w in weights]
        gout = grad_out.contiguous()
        gy = torch.empty(B, L, nx, device=u.device, dtype=torch.float32)
        gas = [torch.empty_like(z) for z in zs]
        gx0 = torch.empty(B, nx, device=u.device, dtype=torch.float32)
        C = _scan_chunks(spec, B, L)
        cl = -(-L // C)
        C = -(-L // cl)
        if C > 1:
            jt = _build_jt(spec, ws, zs)
            pc = torch.empty(B, C, nx, nx, device=u.device)
            sc = torch.empty(B, C, nx, device=u.device)
            bnd = torch.empty(B, C + 1, nx, device=u.device)
            lib.ssm_bwd_summary(jt, gout, pc, sc, L, cl, C, threads=B * C * 32, group_size=32)
            lib.ssm_bwd_scan(pc, sc, bnd, C, threads=B * 32, group_size=32)
        else:
            jt = bnd = torch.zeros(1, device=u.device)
        lib.ssm_bwd(gout, *zs, *ws, jt, bnd, gy, *gas, gx0, L, cl, C, threads=B * C * tpg, group_size=tpg)
        grads, du = mlp_param_grads(spec, x0, u, out, zs, gy, gas, w0=weights[0], need_du=ctx.needs_input_grad[2])
        dx0 = gx0 if ctx.needs_input_grad[3] else None
        return (None, None, du, dx0, *grads)


def metal_rollout(spec: SSMSpec, u: torch.Tensor, x0: torch.Tensor, params: list[torch.Tensor]) -> torch.Tensor:
    """Run the rollout through the generated persistent Metal kernels (autograd-capable)."""
    check_rollout_args(spec, u, x0, "mps")
    if not fits(spec):
        raise RuntimeError(
            f"spec {spec} exceeds the metal backend envelope (layer widths <= {_MAX_WIDTH}, "
            f"<= {_MAX_LINEAR} linear layers); use backend='eager'"
        )
    lib = _get_lib(spec)
    if not torch.is_grad_enabled() or not any(t.requires_grad for t in [u, x0, *params]):
        out, _ = _run_fwd(lib, spec, u.contiguous(), x0.contiguous(), params, store_z=False)
        return out
    return _MetalSSMRollout.apply(lib, spec, u, x0, *params)
