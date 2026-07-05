"""Persistent-kernel Triton backend for the PHNN core: fused GPU rollout and BPTT.

The whole section rollout runs as ONE launch, one program per batch lane, weights
resident on-chip -- the only thing that addresses the kernel-granularity bound of the
compiled loop (~10^5 tiny kernels/step). float32. Applicability caps live in
``common.supports`` (``hidden<=128, n_state<=16, num_layers<=2``). The component-net
weights are repacked/padded on the host into power-of-two tiles (``_prep``), with the
J/R/G output layers reshaped to ``n x n``/``n x m`` so the kernel forms the structure
matrices directly.

Forward stores only the per-step input states (B x L x n). The backward recomputes each
step's intra-step activations and runs the reverse pass of ``MATH.md`` (section 3).
Parameter gradients accumulate into per-lane ``[B, ...]`` buffers (one slice per
program, no atomics) and are summed on the host -- the same split the C backend uses.
"""

__all__ = [
    "triton_rollout",
    "is_available",
]

import torch

from .common import PHNNSpec, bound_value, params_of, supports


def is_available() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        import triton  # noqa: F401
    except ImportError:
        return False
    return True


def _pow2(n: int) -> int:
    return 1 << (n - 1).bit_length()


# ------------------------------------------------------------------ host weight repack
def _pad2(w, pin, pout):
    """Pad an ``[in, out]`` tile to ``[pin, pout]``."""
    out = w.new_zeros(pin, pout)
    out[: w.shape[0], : w.shape[1]] = w
    return out


def _prep(core, spec: PHNNSpec):
    """Repack component-net params into padded kernel tiles (a flat dict of tensors).

    Weight convention: ``[in, out]`` (transpose of ``nn.Linear.weight``). The J/R/G
    output layers are reshaped from a flat ``n²``/``n·m`` vector into a matrix tile so
    the kernel indexes ``B[i,j]``/``G[i,jm]`` directly.
    """
    p = params_of(core)
    n, m, ny, nh = spec.n_state, spec.n_input, spec.n_output, spec.hidden
    pn, pm, pny, ph = _pow2(n), _pow2(m), _pow2(ny), _pow2(nh)
    mid = spec.num_layers == 2
    d = {}

    def mlp(prefix, w, b, out_matrix):
        # in layer: [nh, n] -> [pn, ph]; mid: [nh, nh] -> [ph, ph]
        d[f"{prefix}_win"] = _pad2(w[0].t(), pn, ph).contiguous()
        d[f"{prefix}_bin"] = _pad2(b[0].reshape(1, -1), 1, ph).reshape(ph).contiguous()
        if mid:
            d[f"{prefix}_wmid"] = _pad2(w[1].t(), ph, ph).contiguous()
            d[f"{prefix}_bmid"] = _pad2(b[1].reshape(1, -1), 1, ph).reshape(ph).contiguous()
        wl, bl = w[-1], b[-1]  # [n_out, nh]
        return wl, bl

    # Hamiltonian: scalar output; last weight -> [ph] vector, last bias scalar
    mlp("h", p["hw"], p["hb"], None)
    d["h_wout"] = _pad2(p["hw"][-1].reshape(1, nh).t(), ph, 1).reshape(ph).contiguous()  # [ph]
    d["h_bout"] = p["hb"][-1].reshape(()).float().contiguous()
    # J, R: last weight [n*n, nh] -> tile [ph, pn, pn] indexed [k, i, j]
    for pre, key in (("j", "jw"), ("r", "rw")):
        mlp(pre, p[f"{pre}w"], p[f"{pre}b"], True)
        wl = p[f"{pre}w"][-1].reshape(n, n, nh).permute(2, 0, 1)  # [nh, n, n]
        bl = p[f"{pre}b"][-1].reshape(n, n)
        wo = wl.new_zeros(ph, pn, pn)
        wo[:nh, :n, :n] = wl
        bo = bl.new_zeros(pn, pn)
        bo[:n, :n] = bl
        d[f"{pre}_wout"] = wo.contiguous()
        d[f"{pre}_bout"] = bo.contiguous()
    # G mlp: last weight [n*m, nh] -> [ph, pn, pm]
    mlp("g", p["gw"], p["gb"], True)
    gwl = p["gw"][-1].reshape(n, m, nh).permute(2, 0, 1)
    gbl = p["gb"][-1].reshape(n, m)
    gwo = gwl.new_zeros(ph, pn, pm)
    gwo[:nh, :n, :m] = gwl
    gbo = gbl.new_zeros(pn, pm)
    gbo[:n, :m] = gbl
    d["g_wout"] = gwo.contiguous()
    d["g_bout"] = gbo.contiguous()
    # G linear bypass: weight [n*m, n] -> [pn(in l), pn(i), pm(jm)]
    glw = p["glw"].reshape(n, m, n).permute(2, 0, 1)  # [n(in), n(i), m(jm)]
    glo = glw.new_zeros(pn, pn, pm)
    glo[:n, :n, :m] = glw
    glb = p["glb"].reshape(n, m)
    glbo = glb.new_zeros(pn, pm)
    glbo[:n, :m] = glb
    d["gl_w"] = glo.contiguous()
    d["gl_b"] = glbo.contiguous()
    if spec.output == "linear":
        d["o_w"] = _pad2(p["ow"].t(), pn, pny).contiguous()  # [pn, pny]
        d["o_b"] = _pad2(p["ob"].reshape(1, -1), 1, pny).reshape(pny).contiguous()
    return d, (pn, pm, pny, ph, mid)


def _scalars(core, spec):
    return dict(
        n=spec.n_state, m=spec.n_input, ny=spec.n_output, nh=spec.hidden,
        dt=float(core.dt), out_linear=spec.output == "linear",
        has_bound=spec.has_bound, bound=bound_value(core),
        jr_scale=float(core.jr_scale), g_scale=float(core.g_scale),
    )


# The kernels are defined lazily so importing this module never requires triton.
_KERNELS = None


def _build_kernels():
    global _KERNELS
    if _KERNELS is not None:
        return _KERNELS
    import triton
    import triton.language as tl

    @triton.jit
    def _tanh(x):
        return 2.0 * tl.sigmoid(2.0 * x) - 1.0

    @triton.jit
    def fields(a, hwin, hbin, hwmid, hbmid, hwout, hbout,
               jwin, jbin, jwmid, jbmid, jwout, jbout,
               rwin, rbin, rwmid, rbmid, rwout, rbout,
               gwin, gbin, gwmid, gbmid, gwout, gbout, glw, glb,
               jr_scale, g_scale, bound, HAS_BOUND: tl.constexpr, HAS_MID: tl.constexpr,
               N: tl.constexpr, M: tl.constexpr, PN: tl.constexpr, PM: tl.constexpr, PH: tl.constexpr):
        # tiles are pre-loaded; returns G [PN,PM], dhdx [PN], drift [PN]
        # ---- Hamiltonian value ----
        hz0 = _tanh(tl.sum(a[:, None] * hwin, axis=0) + hbin)  # [PH]
        hz_last = hz0
        if HAS_MID:
            hz1 = _tanh(tl.sum(hz0[:, None] * hwmid, axis=0) + hbmid)
            hz_last = hz1
        h_raw = tl.sum(hz_last * hwout) + hbout  # scalar
        # ---- Hamiltonian gradient tape ----
        gz_last = hwout  # [PH]
        gp_last = gz_last * (1.0 - hz_last * hz_last)
        if HAS_MID:
            gz0 = tl.sum(hwmid * gp_last[None, :], axis=1)  # [PH]
            gp0 = gz0 * (1.0 - hz0 * hz0)
            dhdx_raw = tl.sum(hwin * gp0[None, :], axis=1)  # [PN]
        else:
            dhdx_raw = tl.sum(hwin * gp_last[None, :], axis=1)
        mv = 1.0
        if HAS_BOUND:
            s = h_raw - bound
            mv = tl.where(s > 0, 1.0, tl.exp(s))
        dhdx = mv * dhdx_raw  # [PN]
        # ---- J, R ----
        jz0 = _tanh(tl.sum(a[:, None] * jwin, axis=0) + jbin)
        jl = jz0
        if HAS_MID:
            jl = _tanh(tl.sum(jz0[:, None] * jwmid, axis=0) + jbmid)
        B = (tl.sum(jl[:, None, None] * jwout, axis=0) + jbout) * jr_scale  # [PN,PN]
        rz0 = _tanh(tl.sum(a[:, None] * rwin, axis=0) + rbin)
        rl = rz0
        if HAS_MID:
            rl = _tanh(tl.sum(rz0[:, None] * rwmid, axis=0) + rbmid)
        A = (tl.sum(rl[:, None, None] * rwout, axis=0) + rbout) * jr_scale  # [PN,PN]
        Jm = B - tl.trans(B)
        Rm = tl.sum(A[:, None, :] * A[None, :, :], axis=2)  # A A^T
        JR = Jm - Rm  # [PN,PN]
        drift = tl.sum(JR * dhdx[None, :], axis=1)  # [PN]
        # ---- G ----
        gz0 = _tanh(tl.sum(a[:, None] * gwin, axis=0) + gbin)
        gl = gz0
        if HAS_MID:
            gl = _tanh(tl.sum(gz0[:, None] * gwmid, axis=0) + gbmid)
        Gm = tl.sum(gl[:, None, None] * gwout, axis=0) + gbout  # [PN,PM]
        Glin = tl.sum(a[:, None, None] * glw, axis=0) + glb  # [PN,PM]
        G = (Gm + Glin) * g_scale
        return G, dhdx, drift

    _KERNELS = dict(tl=tl, triton=triton, fields=fields, _tanh=_tanh)
    _build_fwd()
    return _KERNELS


def _build_fwd():
    import triton
    import triton.language as tl

    fields = _KERNELS["fields"]

    @triton.jit
    def load_tiles(hwin_p, hbin_p, hwmid_p, hbmid_p, hwout_p,
                   jwin_p, jbin_p, jwmid_p, jbmid_p, jwout_p, jbout_p,
                   rwin_p, rbin_p, rwmid_p, rbmid_p, rwout_p, rbout_p,
                   gwin_p, gbin_p, gwmid_p, gbmid_p, gwout_p, gbout_p, glw_p, glb_p,
                   PN: tl.constexpr, PM: tl.constexpr, PH: tl.constexpr):
        rn = tl.arange(0, PN)
        rm = tl.arange(0, PM)
        rh = tl.arange(0, PH)
        m2 = rn[:, None] * PH + rh[None, :]      # [PN,PH]
        mh = rh[:, None] * PH + rh[None, :]      # [PH,PH]
        m3n = rh[:, None, None] * (PN * PN) + rn[None, :, None] * PN + rn[None, None, :]  # [PH,PN,PN]
        m3m = rh[:, None, None] * (PN * PM) + rn[None, :, None] * PM + rm[None, None, :]  # [PH,PN,PM]
        mgl = rn[:, None, None] * (PN * PM) + rn[None, :, None] * PM + rm[None, None, :]  # [PN,PN,PM]
        mnn = rn[:, None] * PN + rn[None, :]     # [PN,PN]
        mnm = rn[:, None] * PM + rm[None, :]     # [PN,PM]
        hwin = tl.load(hwin_p + m2)
        hbin = tl.load(hbin_p + rh)
        hwmid = tl.load(hwmid_p + mh)
        hbmid = tl.load(hbmid_p + rh)
        hwout = tl.load(hwout_p + rh)
        jwin = tl.load(jwin_p + m2)
        jbin = tl.load(jbin_p + rh)
        jwmid = tl.load(jwmid_p + mh)
        jbmid = tl.load(jbmid_p + rh)
        jwout = tl.load(jwout_p + m3n)
        jbout = tl.load(jbout_p + mnn)
        rwin = tl.load(rwin_p + m2)
        rbin = tl.load(rbin_p + rh)
        rwmid = tl.load(rwmid_p + mh)
        rbmid = tl.load(rbmid_p + rh)
        rwout = tl.load(rwout_p + m3n)
        rbout = tl.load(rbout_p + mnn)
        gwin = tl.load(gwin_p + m2)
        gbin = tl.load(gbin_p + rh)
        gwmid = tl.load(gwmid_p + mh)
        gbmid = tl.load(gbmid_p + rh)
        gwout = tl.load(gwout_p + m3m)
        gbout = tl.load(gbout_p + mnm)
        glw = tl.load(glw_p + mgl)
        glb = tl.load(glb_p + mnm)
        return (hwin, hbin, hwmid, hbmid, hwout, jwin, jbin, jwmid, jbmid, jwout, jbout,
                rwin, rbin, rwmid, rbmid, rwout, rbout, gwin, gbin, gwmid, gbmid, gwout, gbout, glw, glb)

    @triton.jit
    def phnn_fwd(u_ptr, x0_ptr, out_ptr, xs_ptr,
                 hwin_p, hbin_p, hwmid_p, hbmid_p, hwout_p, hbout,
                 jwin_p, jbin_p, jwmid_p, jbmid_p, jwout_p, jbout_p,
                 rwin_p, rbin_p, rwmid_p, rbmid_p, rwout_p, rbout_p,
                 gwin_p, gbin_p, gwmid_p, gbmid_p, gwout_p, gbout_p, glw_p, glb_p, ow_p, ob_p,
                 jr_scale, g_scale, dt, bound, L,
                 HAS_BOUND: tl.constexpr, HAS_MID: tl.constexpr, OUT_LINEAR: tl.constexpr,
                 N: tl.constexpr, M: tl.constexpr, NY: tl.constexpr,
                 PN: tl.constexpr, PM: tl.constexpr, PNY: tl.constexpr, PH: tl.constexpr):
        pid = tl.program_id(0)
        rn = tl.arange(0, PN)
        rm = tl.arange(0, PM)
        rny = tl.arange(0, PNY)
        (hwin, hbin, hwmid, hbmid, hwout, jwin, jbin, jwmid, jbmid, jwout, jbout,
         rwin, rbin, rwmid, rbmid, rwout, rbout, gwin, gbin, gwmid, gbmid, gwout, gbout, glw, glb) = load_tiles(
            hwin_p, hbin_p, hwmid_p, hbmid_p, hwout_p, jwin_p, jbin_p, jwmid_p, jbmid_p, jwout_p, jbout_p,
            rwin_p, rbin_p, rwmid_p, rbmid_p, rwout_p, rbout_p, gwin_p, gbin_p, gwmid_p, gbmid_p,
            gwout_p, gbout_p, glw_p, glb_p, PN, PM, PH)
        ow = tl.load(ow_p + rn[:, None] * PNY + rny[None, :])
        ob = tl.load(ob_p + rny)
        x = tl.load(x0_ptr + pid * N + rn, mask=rn < N, other=0.0)  # [PN]
        for t in range(0, L):
            uu = tl.load(u_ptr + pid * (L * M) + t * M + rm, mask=rm < M, other=0.0)  # [PM]
            tl.store(xs_ptr + pid * (L * N) + t * N + rn, x, mask=rn < N)
            G, dhdx, drift = fields(x, hwin, hbin, hwmid, hbmid, hwout, hbout, jwin, jbin, jwmid, jbmid, jwout, jbout,
                                    rwin, rbin, rwmid, rbmid, rwout, rbout, gwin, gbin, gwmid, gbmid, gwout, gbout,
                                    glw, glb, jr_scale, g_scale, bound, HAS_BOUND, HAS_MID, N, M, PN, PM, PH)
            if OUT_LINEAR:
                y = tl.sum(x[:, None] * ow, axis=0) + ob  # [PNY]
                tl.store(out_ptr + pid * (L * NY) + t * NY + rny, y, mask=rny < NY)
            else:
                y = tl.sum(G * dhdx[:, None], axis=0)  # [PM] = G^T dhdx
                tl.store(out_ptr + pid * (L * NY) + t * NY + rm, y, mask=rm < NY)
            k1 = dt * (drift + tl.sum(G * uu[None, :], axis=1))
            G2, _, drift2 = fields(x + k1 / 2, hwin, hbin, hwmid, hbmid, hwout, hbout, jwin, jbin, jwmid, jbmid, jwout,
                                   jbout, rwin, rbin, rwmid, rbmid, rwout, rbout, gwin, gbin, gwmid, gbmid, gwout, gbout,
                                   glw, glb, jr_scale, g_scale, bound, HAS_BOUND, HAS_MID, N, M, PN, PM, PH)
            k2 = dt * (drift2 + tl.sum(G2 * uu[None, :], axis=1))
            G3, _, drift3 = fields(x + k2 / 2, hwin, hbin, hwmid, hbmid, hwout, hbout, jwin, jbin, jwmid, jbmid, jwout,
                                   jbout, rwin, rbin, rwmid, rbmid, rwout, rbout, gwin, gbin, gwmid, gbmid, gwout, gbout,
                                   glw, glb, jr_scale, g_scale, bound, HAS_BOUND, HAS_MID, N, M, PN, PM, PH)
            k3 = dt * (drift3 + tl.sum(G3 * uu[None, :], axis=1))
            G4, _, drift4 = fields(x + k3, hwin, hbin, hwmid, hbmid, hwout, hbout, jwin, jbin, jwmid, jbmid, jwout,
                                   jbout, rwin, rbin, rwmid, rbmid, rwout, rbout, gwin, gbin, gwmid, gbmid, gwout, gbout,
                                   glw, glb, jr_scale, g_scale, bound, HAS_BOUND, HAS_MID, N, M, PN, PM, PH)
            k4 = dt * (drift4 + tl.sum(G4 * uu[None, :], axis=1))
            x = x + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    _KERNELS["phnn_fwd"] = phnn_fwd
    _KERNELS["load_tiles"] = load_tiles
    _build_bwd()


def _build_bwd():
    import triton
    import triton.language as tl

    load_tiles = _KERNELS["load_tiles"]
    _tanh = _KERNELS["_tanh"]

    @triton.jit
    def _fields3(a, tiles, hbout, jr_scale, g_scale, bound, HAS_BOUND: tl.constexpr, HAS_MID: tl.constexpr,
                 N: tl.constexpr, M: tl.constexpr, PN: tl.constexpr, PM: tl.constexpr, PH: tl.constexpr):
        (hwin, hbin, hwmid, hbmid, hwout, jwin, jbin, jwmid, jbmid, jwout, jbout,
         rwin, rbin, rwmid, rbmid, rwout, rbout, gwin, gbin, gwmid, gbmid, gwout, gbout, glw, glb) = tiles
        hz0 = _tanh(tl.sum(a[:, None] * hwin, axis=0) + hbin)
        zl = hz0
        if HAS_MID:
            zl = _tanh(tl.sum(hz0[:, None] * hwmid, axis=0) + hbmid)
        h_raw = tl.sum(zl * hwout) + hbout
        gp_last = hwout * (1.0 - zl * zl)
        if HAS_MID:
            gz0 = tl.sum(hwmid * gp_last[None, :], axis=1)
            gp0 = gz0 * (1.0 - hz0 * hz0)
            dhdx_raw = tl.sum(hwin * gp0[None, :], axis=1)
        else:
            dhdx_raw = tl.sum(hwin * gp_last[None, :], axis=1)
        mv = 1.0
        if HAS_BOUND:
            s = h_raw - bound
            mv = tl.where(s > 0, 1.0, tl.exp(s))
        dhdx = mv * dhdx_raw
        jz0 = _tanh(tl.sum(a[:, None] * jwin, axis=0) + jbin)
        jl = jz0
        if HAS_MID:
            jl = _tanh(tl.sum(jz0[:, None] * jwmid, axis=0) + jbmid)
        B = (tl.sum(jl[:, None, None] * jwout, axis=0) + jbout) * jr_scale
        rz0 = _tanh(tl.sum(a[:, None] * rwin, axis=0) + rbin)
        rl = rz0
        if HAS_MID:
            rl = _tanh(tl.sum(rz0[:, None] * rwmid, axis=0) + rbmid)
        A = (tl.sum(rl[:, None, None] * rwout, axis=0) + rbout) * jr_scale
        JR = (B - tl.trans(B)) - tl.sum(A[:, None, :] * A[None, :, :], axis=2)
        drift = tl.sum(JR * dhdx[None, :], axis=1)
        gz0 = _tanh(tl.sum(a[:, None] * gwin, axis=0) + gbin)
        gl = gz0
        if HAS_MID:
            gl = _tanh(tl.sum(gz0[:, None] * gwmid, axis=0) + gbmid)
        Gm = tl.sum(gl[:, None, None] * gwout, axis=0) + gbout
        Glin = tl.sum(a[:, None, None] * glw, axis=0) + glb
        G = (Gm + Glin) * g_scale
        return G, dhdx, drift

    @triton.jit
    def acc1(ptr, pid, delta, PA: tl.constexpr):
        r = tl.arange(0, PA)
        off = pid * PA + r
        tl.store(ptr + off, tl.load(ptr + off) + delta)

    @triton.jit
    def acc2(ptr, pid, delta, PA: tl.constexpr, PB: tl.constexpr):
        ra = tl.arange(0, PA)
        rb = tl.arange(0, PB)
        off = pid * (PA * PB) + ra[:, None] * PB + rb[None, :]
        tl.store(ptr + off, tl.load(ptr + off) + delta)

    @triton.jit
    def acc3(ptr, pid, delta, PA: tl.constexpr, PB: tl.constexpr, PC: tl.constexpr):
        ra = tl.arange(0, PA)
        rb = tl.arange(0, PB)
        rc = tl.arange(0, PC)
        off = pid * (PA * PB * PC) + ra[:, None, None] * (PB * PC) + rb[None, :, None] * PC + rc[None, None, :]
        tl.store(ptr + off, tl.load(ptr + off) + delta)

    @triton.jit
    def mlp_vjp(a, z0, z1, win, wmid, wout, bout, dwin_p, dbin_p, dwmid_p, dbmid_p, dwout_p, dbout_p, pid,
                HAS_MID: tl.constexpr, PN: tl.constexpr, PH: tl.constexpr, POUT: tl.constexpr):
        # linear (matrix) output layer: out[i,j] = sum_k zl[k]*wout[k,i,j] + bias
        zl = z1 if HAS_MID else z0
        acc3(dwout_p, pid, zl[:, None, None] * bout[None, :, :], PH, PN, POUT)
        acc2(dbout_p, pid, bout, PN, POUT)
        bzl = tl.sum(tl.sum(wout * bout[None, :, :], axis=2), axis=1)  # [PH]
        if HAS_MID:
            bp1 = bzl * (1.0 - z1 * z1)
            acc2(dwmid_p, pid, z0[:, None] * bp1[None, :], PH, PH)
            acc1(dbmid_p, pid, bp1, PH)
            bz0 = tl.sum(wmid * bp1[None, :], axis=1)
            bp0 = bz0 * (1.0 - z0 * z0)
        else:
            bp0 = bzl * (1.0 - z0 * z0)
        acc2(dwin_p, pid, a[:, None] * bp0[None, :], PN, PH)
        acc1(dbin_p, pid, bp0, PH)
        return tl.sum(win * bp0[None, :], axis=1)  # bq [PN]

    @triton.jit
    def hnet_vjp(a, z0, z1, win, wmid, wout, b_dhdx_raw, b_Hraw,
                 dwin_p, dbin_p, dwmid_p, dbmid_p, dwout_p, dbout_p, pid,
                 HAS_MID: tl.constexpr, PN: tl.constexpr, PH: tl.constexpr):
        zl = z1 if HAS_MID else z0
        gp_last = wout * (1.0 - zl * zl)  # gz_last = wout
        if HAS_MID:
            gz0 = tl.sum(wmid * gp_last[None, :], axis=1)
            gp0 = gz0 * (1.0 - z0 * z0)
        else:
            gp0 = gp_last
        # gradient tape reverse
        acc2(dwin_p, pid, b_dhdx_raw[:, None] * gp0[None, :], PN, PH)
        bgp0 = tl.sum(win * b_dhdx_raw[:, None], axis=0)  # [PH]
        bgz0 = bgp0 * (1.0 - z0 * z0)
        bd0 = bgp0 * (gz0 if HAS_MID else wout)
        if HAS_MID:
            acc2(dwmid_p, pid, bgz0[:, None] * gp_last[None, :], PH, PH)
            bgp1 = tl.sum(wmid * bgz0[:, None], axis=0)
            bgz1 = bgp1 * (1.0 - z1 * z1)
            bd1 = bgp1 * wout
            acc1(dwout_p, pid, bgz1, PH)
            # value tape
            bz1 = wout * b_Hraw + (-2.0 * z1 * bd1)
            acc1(dwout_p, pid, b_Hraw * z1, PH)
            acc1(dbout_p, pid, b_Hraw, 1)
            bp1 = bz1 * (1.0 - z1 * z1)
            acc2(dwmid_p, pid, z0[:, None] * bp1[None, :], PH, PH)
            acc1(dbmid_p, pid, bp1, PH)
            bz0 = tl.sum(wmid * bp1[None, :], axis=1) + (-2.0 * z0 * bd0)
        else:
            acc1(dwout_p, pid, bgz0, PH)
            bz0 = wout * b_Hraw + (-2.0 * z0 * bd0)
            acc1(dwout_p, pid, b_Hraw * z0, PH)
            acc1(dbout_p, pid, b_Hraw, 1)
        bp0 = bz0 * (1.0 - z0 * z0)
        acc2(dwin_p, pid, a[:, None] * bp0[None, :], PN, PH)
        acc1(dbin_p, pid, bp0, PH)
        return tl.sum(win * bp0[None, :], axis=1)  # bq [PN]

    @triton.jit
    def fields_vjp(a, tiles, hbout, gptrs, bG, bdhdx_ext, bdrift, pid,
                   jr_scale, g_scale, bound, HAS_BOUND: tl.constexpr, HAS_MID: tl.constexpr,
                   N: tl.constexpr, M: tl.constexpr, PN: tl.constexpr, PM: tl.constexpr, PH: tl.constexpr):
        (hwin, hbin, hwmid, hbmid, hwout, jwin, jbin, jwmid, jbmid, jwout, jbout,
         rwin, rbin, rwmid, rbmid, rwout, rbout, gwin, gbin, gwmid, gbmid, gwout, gbout, glw, glb) = tiles
        (dhwin, dhbin, dhwmid, dhbmid, dhwout, dhbout, djwin, djbin, djwmid, djbmid, djwout, djbout,
         drwin, drbin, drwmid, drbmid, drwout, drbout, dgwin, dgbin, dgwmid, dgbmid, dgwout, dgbout,
         dglw, dglb) = gptrs
        # recompute forward activations at a
        hz0 = _tanh(tl.sum(a[:, None] * hwin, axis=0) + hbin)
        hz1 = hz0
        if HAS_MID:
            hz1 = _tanh(tl.sum(hz0[:, None] * hwmid, axis=0) + hbmid)
        zl = hz1 if HAS_MID else hz0
        h_raw = tl.sum(zl * hwout) + hbout
        gp_last = hwout * (1.0 - zl * zl)
        if HAS_MID:
            gz0h = tl.sum(hwmid * gp_last[None, :], axis=1)
            gp0h = gz0h * (1.0 - hz0 * hz0)
            dhdx_raw = tl.sum(hwin * gp0h[None, :], axis=1)
        else:
            dhdx_raw = tl.sum(hwin * gp_last[None, :], axis=1)
        mv = 1.0
        if HAS_BOUND:
            s = h_raw - bound
            mv = tl.where(s > 0, 1.0, tl.exp(s))
        dhdx = mv * dhdx_raw
        jz0 = _tanh(tl.sum(a[:, None] * jwin, axis=0) + jbin)
        jl = jz0
        if HAS_MID:
            jl = _tanh(tl.sum(jz0[:, None] * jwmid, axis=0) + jbmid)
        rz0 = _tanh(tl.sum(a[:, None] * rwin, axis=0) + rbin)
        rl = rz0
        if HAS_MID:
            rl = _tanh(tl.sum(rz0[:, None] * rwmid, axis=0) + rbmid)
        A = (tl.sum(rl[:, None, None] * rwout, axis=0) + rbout) * jr_scale  # [PN,PN]
        gz0 = _tanh(tl.sum(a[:, None] * gwin, axis=0) + gbin)
        gl = gz0
        if HAS_MID:
            gl = _tanh(tl.sum(gz0[:, None] * gwmid, axis=0) + gbmid)
        # JR^T bdrift and bJR
        Jm = (tl.sum(jl[:, None, None] * jwout, axis=0) + jbout) * jr_scale
        Jm = Jm - tl.trans(Jm)
        Rm = tl.sum(A[:, None, :] * A[None, :, :], axis=2)
        JR = Jm - Rm  # [PN,PN]
        bJR = bdrift[:, None] * dhdx[None, :]
        bdhdx = bdhdx_ext + tl.sum(JR * bdrift[:, None], axis=0)
        bjout = jr_scale * (bJR - tl.trans(bJR))
        bJRs = bJR + tl.trans(bJR)
        bA = -tl.sum(bJRs[:, :, None] * A[None, :, :], axis=1)  # [i,k]=sum_j bJRs[i,j]A[j,k]
        brout = jr_scale * bA
        bgout = g_scale * bG
        b_dhdx_raw = mv * bdhdx
        b_Hraw = 0.0
        if HAS_BOUND:
            bm = tl.sum(dhdx_raw * bdhdx)
            b_Hraw = bm * tl.where(s > 0, 0.0, tl.exp(s))
        bq = hnet_vjp(a, hz0, hz1, hwin, hwmid, hwout, b_dhdx_raw, b_Hraw,
                      dhwin, dhbin, dhwmid, dhbmid, dhwout, dhbout, pid, HAS_MID, PN, PH)
        bq += mlp_vjp(a, jz0, jl, jwin, jwmid, jwout, bjout, djwin, djbin, djwmid, djbmid, djwout, djbout,
                      pid, HAS_MID, PN, PH, PN)
        bq += mlp_vjp(a, rz0, rl, rwin, rwmid, rwout, brout, drwin, drbin, drwmid, drbmid, drwout, drbout,
                      pid, HAS_MID, PN, PH, PN)
        # g-net linear bypass
        acc3(dglw, pid, a[:, None, None] * bgout[None, :, :], PN, PN, PM)
        acc2(dglb, pid, bgout, PN, PM)
        bq += tl.sum(tl.sum(glw * bgout[None, :, :], axis=2), axis=1)  # Wlin^T bgout [PN]
        bq += mlp_vjp(a, gz0, gl, gwin, gwmid, gwout, bgout, dgwin, dgbin, dgwmid, dgbmid, dgwout, dgbout,
                      pid, HAS_MID, PN, PH, PM)
        return bq

    @triton.jit
    def phnn_bwd(u_ptr, xs_ptr, go_ptr, du_ptr, gx0_ptr,
                 hwin_p, hbin_p, hwmid_p, hbmid_p, hwout_p, hbout,
                 jwin_p, jbin_p, jwmid_p, jbmid_p, jwout_p, jbout_p,
                 rwin_p, rbin_p, rwmid_p, rbmid_p, rwout_p, rbout_p,
                 gwin_p, gbin_p, gwmid_p, gbmid_p, gwout_p, gbout_p, glw_p, glb_p, ow_p, ob_p,
                 dhwin, dhbin, dhwmid, dhbmid, dhwout, dhbout,
                 djwin, djbin, djwmid, djbmid, djwout, djbout,
                 drwin, drbin, drwmid, drbmid, drwout, drbout,
                 dgwin, dgbin, dgwmid, dgbmid, dgwout, dgbout, dglw, dglb, dow, dob,
                 jr_scale, g_scale, dt, bound, L,
                 HAS_BOUND: tl.constexpr, HAS_MID: tl.constexpr, OUT_LINEAR: tl.constexpr,
                 N: tl.constexpr, M: tl.constexpr, NY: tl.constexpr,
                 PN: tl.constexpr, PM: tl.constexpr, PNY: tl.constexpr, PH: tl.constexpr):
        pid = tl.program_id(0)
        rn = tl.arange(0, PN)
        rm = tl.arange(0, PM)
        rny = tl.arange(0, PNY)
        tiles = load_tiles(hwin_p, hbin_p, hwmid_p, hbmid_p, hwout_p, jwin_p, jbin_p, jwmid_p, jbmid_p, jwout_p,
                           jbout_p, rwin_p, rbin_p, rwmid_p, rbmid_p, rwout_p, rbout_p, gwin_p, gbin_p, gwmid_p,
                           gbmid_p, gwout_p, gbout_p, glw_p, glb_p, PN, PM, PH)
        gptrs = (dhwin, dhbin, dhwmid, dhbmid, dhwout, dhbout, djwin, djbin, djwmid, djbmid, djwout, djbout,
                 drwin, drbin, drwmid, drbmid, drwout, drbout, dgwin, dgbin, dgwmid, dgbmid, dgwout, dgbout,
                 dglw, dglb)
        ow = tl.load(ow_p + rn[:, None] * PNY + rny[None, :])

        bx = tl.zeros([PN], dtype=tl.float32)
        for ti in range(0, L):
            t = L - 1 - ti
            xt = tl.load(xs_ptr + pid * (L * N) + t * N + rn, mask=rn < N, other=0.0)
            uu = tl.load(u_ptr + pid * (L * M) + t * M + rm, mask=rm < M, other=0.0)
            gy = tl.load(go_ptr + pid * (L * NY) + t * NY + rny, mask=rny < NY, other=0.0)  # [PNY]
            # recompute k1..k4 (need drift,G per stage) via fields (forward)
            g1, d1, dr1 = _fields3(xt, tiles, hbout, jr_scale, g_scale, bound, HAS_BOUND, HAS_MID, N, M, PN, PM, PH)
            k1 = dt * (dr1 + tl.sum(g1 * uu[None, :], axis=1))
            g2, _, dr2 = _fields3(xt + k1 / 2, tiles, hbout, jr_scale, g_scale, bound, HAS_BOUND, HAS_MID, N, M, PN, PM, PH)
            k2 = dt * (dr2 + tl.sum(g2 * uu[None, :], axis=1))
            g3, _, dr3 = _fields3(xt + k2 / 2, tiles, hbout, jr_scale, g_scale, bound, HAS_BOUND, HAS_MID, N, M, PN, PM, PH)
            k3 = dt * (dr3 + tl.sum(g3 * uu[None, :], axis=1))
            g4, _, _ = _fields3(xt + k3, tiles, hbout, jr_scale, g_scale, bound, HAS_BOUND, HAS_MID, N, M, PN, PM, PH)
            q2 = xt + k1 / 2
            q3 = xt + k2 / 2
            q4 = xt + k3
            # reverse
            bxin = bx
            bk1 = bx / 6
            bk2 = bx / 3
            bk3 = bx / 3
            bk4 = bx / 6
            bG1 = tl.zeros([PN, PM], dtype=tl.float32)
            bdhdx_ext = tl.zeros([PN], dtype=tl.float32)
            if OUT_LINEAR:
                bxin += tl.sum(ow * gy[None, :], axis=1)  # Wout^T gy ; ow [PN,PNY]
                acc2(dow, pid, xt[:, None] * gy[None, :], PN, PNY)
                acc1(dob, pid, gy, PNY)
            else:
                # y = G1^T dhdx1 ; gy is [PM]
                bG1 += d1[:, None] * gy[None, :]  # [PN,PM]: dhdx1[i]*gy[j]
                bdhdx_ext += tl.sum(g1 * gy[None, :], axis=1)  # G1 gy
            # stage 4
            br4 = dt * bk4
            acc_du4 = tl.sum(g4 * br4[:, None], axis=0)  # G4^T br4 [PM]
            bq4 = fields_vjp(q4, tiles, hbout, gptrs, br4[:, None] * uu[None, :], tl.zeros([PN], dtype=tl.float32), br4,
                             pid, jr_scale, g_scale, bound, HAS_BOUND, HAS_MID, N, M, PN, PM, PH)
            bxin += bq4
            bk3 += bq4
            # stage 3
            br3 = dt * bk3
            acc_du3 = tl.sum(g3 * br3[:, None], axis=0)
            bq3 = fields_vjp(q3, tiles, hbout, gptrs, br3[:, None] * uu[None, :], tl.zeros([PN], dtype=tl.float32), br3,
                             pid, jr_scale, g_scale, bound, HAS_BOUND, HAS_MID, N, M, PN, PM, PH)
            bxin += bq3
            bk2 += bq3 / 2
            # stage 2
            br2 = dt * bk2
            acc_du2 = tl.sum(g2 * br2[:, None], axis=0)
            bq2 = fields_vjp(q2, tiles, hbout, gptrs, br2[:, None] * uu[None, :], tl.zeros([PN], dtype=tl.float32), br2,
                             pid, jr_scale, g_scale, bound, HAS_BOUND, HAS_MID, N, M, PN, PM, PH)
            bxin += bq2
            bk1 += bq2 / 2
            # stage 1 (with output adjoints)
            br1 = dt * bk1
            acc_du1 = tl.sum(g1 * br1[:, None], axis=0)
            bG1 += br1[:, None] * uu[None, :]
            bq1 = fields_vjp(xt, tiles, hbout, gptrs, bG1, bdhdx_ext, br1,
                             pid, jr_scale, g_scale, bound, HAS_BOUND, HAS_MID, N, M, PN, PM, PH)
            bxin += bq1
            du_t = acc_du1 + acc_du2 + acc_du3 + acc_du4
            tl.store(du_ptr + pid * (L * M) + t * M + rm, du_t, mask=rm < M)
            bx = bxin
        tl.store(gx0_ptr + pid * N + rn, bx, mask=rn < N)

    _KERNELS["phnn_bwd"] = phnn_bwd


# kernel pointer-arg order (h_bout is a scalar, passed separately)
_ARG_KEYS = [
    "h_win", "h_bin", "h_wmid", "h_bmid", "h_wout",
    "j_win", "j_bin", "j_wmid", "j_bmid", "j_wout", "j_bout",
    "r_win", "r_bin", "r_wmid", "r_bmid", "r_wout", "r_bout",
    "g_win", "g_bin", "g_wmid", "g_bmid", "g_wout", "g_bout", "gl_w", "gl_b",
]


def _weight_args(d, ph):
    """Ordered weight tensors for the kernels
    fill absent mid layers with dummies."""
    dev = d["h_win"].device
    dummy_v = torch.zeros(ph, device=dev)
    dummy_m = torch.zeros(ph, ph, device=dev)

    def get(k):
        if k in d:
            return d[k]
        return dummy_m if k.endswith("wmid") else dummy_v

    return [get(k) for k in _ARG_KEYS]


def _num_warps(spec):
    # small per-layer GEMVs: fewer warps win (less register spill across the many resident tiles)
    return 2 if spec.hidden <= 64 else 4


def _run_fwd(core, spec, u, x0):
    K = _build_kernels()
    d, (pn, pm, pny, ph, mid) = _prep(core, spec)
    sc = _scalars(core, spec)
    B, L = u.shape[0], u.shape[1]
    dev = u.device
    out = torch.empty(B, L, spec.n_output, device=dev, dtype=torch.float32)
    xs = torch.empty(B, L, spec.n_state, device=dev, dtype=torch.float32)
    wargs = _weight_args(d, ph)
    hbout = float(d["h_bout"])
    ow = d.get("o_w", torch.zeros(pn, pny, device=dev))
    ob = d.get("o_b", torch.zeros(pny, device=dev))
    K["phnn_fwd"][(B,)](
        u.contiguous(), x0.contiguous(), out, xs,
        wargs[0], wargs[1], wargs[2], wargs[3], wargs[4], hbout, *wargs[5:], ow, ob,
        sc["jr_scale"], sc["g_scale"], sc["dt"], sc["bound"], L,
        sc["has_bound"], mid, sc["out_linear"],
        sc["n"], sc["m"], sc["ny"], pn, pm, pny, ph, num_warps=_num_warps(spec),
    )
    return out, xs


def _zeros(*shape, dev):
    return torch.zeros(*shape, device=dev, dtype=torch.float32)


def _grad_buffers(spec, pn, pm, pny, ph, mid, B, dev):
    """Per-lane ``[B, ...]`` grad buffers matching the padded kernel tiles, in kernel-arg order."""
    z = lambda *s: _zeros(B, *s, dev=dev)  # noqa: E731
    hb = [z(pn, ph), z(ph), z(ph, ph), z(ph), z(ph), z(1)]  # win,bin,wmid,bmid,wout,bout(scalar)
    jb = [z(pn, ph), z(ph), z(ph, ph), z(ph), z(ph, pn, pn), z(pn, pn)]
    rb = [z(pn, ph), z(ph), z(ph, ph), z(ph), z(ph, pn, pn), z(pn, pn)]
    gb = [z(pn, ph), z(ph), z(ph, ph), z(ph), z(ph, pn, pm), z(pn, pm)]
    gl = [z(pn, pn, pm), z(pn, pm)]
    ob = [z(pn, pny), z(pny)]
    return hb, jb, rb, gb, gl, ob


def _gather(spec, bufs, mid):
    """Reduce per-lane padded grad buffers to parameter grads in ``flat_params`` order."""
    n, m, ny, nh = spec.n_state, spec.n_input, spec.n_output, spec.hidden
    hb, jb, rb, gb, gl, ob = bufs

    def mlp_grads(b, out_matrix, out_cols):
        # b = [win,bin,wmid,bmid,wout,bout]
        gw = [b[0].sum(0)[:n, :nh].t().contiguous()]  # w0 [nh,n]
        gbi = [b[1].sum(0)[:nh].contiguous()]
        if mid:
            gw.append(b[2].sum(0)[:nh, :nh].t().contiguous())
            gbi.append(b[3].sum(0)[:nh].contiguous())
        if out_matrix:
            gw.append(b[4].sum(0)[:nh, :n, :out_cols].permute(1, 2, 0).reshape(n * out_cols, nh).contiguous())
            gbi.append(b[5].sum(0)[:n, :out_cols].reshape(n * out_cols).contiguous())
        return gw, gbi

    hw = [hb[0].sum(0)[:n, :nh].t().contiguous()]
    hbi = [hb[1].sum(0)[:nh].contiguous()]
    if mid:
        hw.append(hb[2].sum(0)[:nh, :nh].t().contiguous())
        hbi.append(hb[3].sum(0)[:nh].contiguous())
    hw.append(hb[4].sum(0)[:nh].reshape(1, nh).contiguous())  # w_last [1,nh]
    hbi.append(hb[5].sum(0).reshape(1).contiguous())  # b_last [1]
    jw, jbi = mlp_grads(jb, True, n)
    rw, rbi = mlp_grads(rb, True, n)
    gw, gbi = mlp_grads(gb, True, m)
    dglw = gl[0].sum(0)[:n, :n, :m].permute(1, 2, 0).reshape(n * m, n).contiguous()
    dglb = gl[1].sum(0)[:n, :m].reshape(n * m).contiguous()
    grads = [*hw, *hbi, *jw, *jbi, *rw, *rbi, dglw, dglb, *gw, *gbi]
    if spec.output == "linear":
        grads += [ob[0].sum(0)[:n, :ny].t().contiguous(), ob[1].sum(0)[:ny].contiguous()]
    return grads


def _run_bwd(core, spec, u, xs, grad_out):
    K = _build_kernels()
    d, (pn, pm, pny, ph, mid) = _prep(core, spec)
    sc = _scalars(core, spec)
    B, L = xs.shape[0], xs.shape[1]
    dev = u.device
    wargs = _weight_args(d, ph)
    hbout = float(d["h_bout"])
    ow = d.get("o_w", _zeros(pn, pny, dev=dev))
    ob = d.get("o_b", _zeros(pny, dev=dev))
    bufs = _grad_buffers(spec, pn, pm, pny, ph, mid, B, dev)
    hb, jb, rb, gb, gl, ob_g = bufs
    du = _zeros(B, L, spec.n_input, dev=dev)
    gx0 = _zeros(B, spec.n_state, dev=dev)
    K["phnn_bwd"][(B,)](
        u.contiguous(), xs.contiguous(), grad_out.contiguous(), du, gx0,
        wargs[0], wargs[1], wargs[2], wargs[3], wargs[4], hbout, *wargs[5:], ow, ob,
        *hb, *jb, *rb, *gb, gl[0], gl[1], ob_g[0], ob_g[1],
        sc["jr_scale"], sc["g_scale"], sc["dt"], sc["bound"], L,
        sc["has_bound"], mid, sc["out_linear"],
        sc["n"], sc["m"], sc["ny"], pn, pm, pny, ph, num_warps=_num_warps(spec),
    )
    grads = _gather(spec, bufs, mid)
    return grads, du, gx0


class _TritonPHNNRollout(torch.autograd.Function):
    @staticmethod
    def forward(ctx, core, spec, u, x0, *params):
        out, xs = _run_fwd(core, spec, u.contiguous(), x0.contiguous())
        ctx.core, ctx.spec = core, spec
        ctx.save_for_backward(u, xs)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        u, xs = ctx.saved_tensors
        grads, du, gx0 = _run_bwd(ctx.core, ctx.spec, u, xs, grad_out.contiguous())
        du_out = du if ctx.needs_input_grad[2] else None
        dx0_out = gx0 if ctx.needs_input_grad[3] else None
        return (None, None, du_out, dx0_out, *grads)


def triton_rollout(core, spec: PHNNSpec, u: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
    """Run the PHNN section rollout through the persistent Triton kernels (autograd-capable)."""
    if not supports(spec, "triton"):
        raise RuntimeError(f"spec {spec} outside the triton envelope")
    if u.dtype != torch.float32:
        raise RuntimeError(f"the triton backend requires float32, got {u.dtype}")
    from .common import flat_params

    params = flat_params(core)
    if not torch.is_grad_enabled() or not any(t.requires_grad for t in [u, x0, *params]):
        out, _ = _run_fwd(core, spec, u.contiguous(), x0.contiguous())
        return out
    return _TritonPHNNRollout.apply(core, spec, u.contiguous(), x0.contiguous(), *params)
