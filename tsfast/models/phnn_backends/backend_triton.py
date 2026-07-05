"""Persistent-kernel Triton backend for the PHNN core: fused GPU rollout, split BPTT.

The whole section rollout runs as ONE launch, one program per batch lane, weights held
as on-chip tiles -- the only thing that addresses the kernel-granularity bound of the
compiled loop (~10^5 tiny kernels/step). float32. Applicability caps live in
``common.supports``. The component-net weights are repacked/padded on the host into
power-of-two tiles (``_prep``), with the J/R/G output layers reshaped to ``n x n``/
``n x m`` so the kernel forms the structure matrices directly.

Backward follows the split-BPTT design of the SSM/NARX backends (see
``ssm/backend_triton.py``): the training forward additionally stores the per-stage
hidden activations, stage states, and ELU-bound arguments; the reverse-sweep kernel
carries only the state adjoint -- it recomputes nothing through tanh, loads the stored
activations, EMITS the per-stage output-layer/Hamiltonian adjoint seeds to global
memory, and keeps just the sequential ``bq`` chains needed for the ``bx`` carry and
``du``. All parameter gradients are then batched cuBLAS GEMMs over the ``[B*L*4, .]``
flattened seeds on the host (``_param_grads``), following ``MATH.md`` (section 3).
This removes the per-lane gradient tiles and their read-modify-write traffic from the
kernel, which previously spilled catastrophically (4346 spill slots) and made ptxas
intractable at ``hidden=128``.
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

    def mlp(prefix, w, b):
        # in layer: [nh, n] -> [pn, ph]; mid: [nh, nh] -> [ph, ph]
        d[f"{prefix}_win"] = _pad2(w[0].t(), pn, ph).contiguous()
        d[f"{prefix}_bin"] = _pad2(b[0].reshape(1, -1), 1, ph).reshape(ph).contiguous()
        if mid:
            d[f"{prefix}_wmid"] = _pad2(w[1].t(), ph, ph).contiguous()
            d[f"{prefix}_bmid"] = _pad2(b[1].reshape(1, -1), 1, ph).reshape(ph).contiguous()

    # Hamiltonian: scalar output; last weight -> [ph] vector, last bias scalar
    mlp("h", p["hw"], p["hb"])
    d["h_wout"] = _pad2(p["hw"][-1].reshape(1, nh).t(), ph, 1).reshape(ph).contiguous()  # [ph]
    d["h_bout"] = p["hb"][-1].detach().reshape(()).float().contiguous()
    # J, R: last weight [n*n, nh] -> tile [ph, pn, pn] indexed [k, i, j]
    for pre in ("j", "r"):
        mlp(pre, p[f"{pre}w"], p[f"{pre}b"])
        wl = p[f"{pre}w"][-1].reshape(n, n, nh).permute(2, 0, 1)  # [nh, n, n]
        bl = p[f"{pre}b"][-1].reshape(n, n)
        wo = wl.new_zeros(ph, pn, pn)
        wo[:nh, :n, :n] = wl
        bo = bl.new_zeros(pn, pn)
        bo[:n, :n] = bl
        d[f"{pre}_wout"] = wo.contiguous()
        d[f"{pre}_bout"] = bo.contiguous()
    # G mlp: last weight [n*m, nh] -> [ph, pn, pm]
    mlp("g", p["gw"], p["gb"])
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
    def fields(a, tiles, hbout, jr_scale, g_scale, bound,
               HAS_BOUND: tl.constexpr, HAS_MID: tl.constexpr,
               PN: tl.constexpr, PM: tl.constexpr, PH: tl.constexpr):
        """Full field evaluation at ``a``: returns everything the forward and the
        activation stores need. ``sval`` is the ELU argument ``H_raw - b`` (0 without
        bound); z1 duplicates z0 when there is no mid layer."""
        (hwin, hbin, hwmid, hbmid, hwout, jwin, jbin, jwmid, jbmid, jwout, jbout,
         rwin, rbin, rwmid, rbmid, rwout, rbout, gwin, gbin, gwmid, gbmid, gwout, gbout, glw, glb) = tiles
        # ---- Hamiltonian value + gradient tape ----
        hz0 = _tanh(tl.sum(a[:, None] * hwin, axis=0) + hbin)  # [PH]
        hz1 = hz0
        if HAS_MID:
            hz1 = _tanh(tl.sum(hz0[:, None] * hwmid, axis=0) + hbmid)
        zl = hz1 if HAS_MID else hz0
        h_raw = tl.sum(zl * hwout) + hbout  # scalar
        gp_last = hwout * (1.0 - zl * zl)
        if HAS_MID:
            gz0 = tl.sum(hwmid * gp_last[None, :], axis=1)  # [PH]
            gp0 = gz0 * (1.0 - hz0 * hz0)
            dhdx_raw = tl.sum(hwin * gp0[None, :], axis=1)  # [PN]
        else:
            dhdx_raw = tl.sum(hwin * gp_last[None, :], axis=1)
        mv = 1.0
        sval = 0.0
        if HAS_BOUND:
            sval = h_raw - bound
            mv = tl.where(sval > 0, 1.0, tl.exp(sval))
        dhdx = mv * dhdx_raw  # [PN]
        # ---- J, R ----
        jz0 = _tanh(tl.sum(a[:, None] * jwin, axis=0) + jbin)
        jz1 = jz0
        if HAS_MID:
            jz1 = _tanh(tl.sum(jz0[:, None] * jwmid, axis=0) + jbmid)
        jl = jz1 if HAS_MID else jz0
        B = (tl.sum(jl[:, None, None] * jwout, axis=0) + jbout) * jr_scale  # [PN,PN]
        rz0 = _tanh(tl.sum(a[:, None] * rwin, axis=0) + rbin)
        rz1 = rz0
        if HAS_MID:
            rz1 = _tanh(tl.sum(rz0[:, None] * rwmid, axis=0) + rbmid)
        rl = rz1 if HAS_MID else rz0
        A = (tl.sum(rl[:, None, None] * rwout, axis=0) + rbout) * jr_scale  # [PN,PN]
        Jm = B - tl.trans(B)
        Rm = tl.sum(A[:, None, :] * A[None, :, :], axis=2)  # A A^T
        JR = Jm - Rm  # [PN,PN]
        drift = tl.sum(JR * dhdx[None, :], axis=1)  # [PN]
        # ---- G ----
        gz0 = _tanh(tl.sum(a[:, None] * gwin, axis=0) + gbin)
        gz1 = gz0
        if HAS_MID:
            gz1 = _tanh(tl.sum(gz0[:, None] * gwmid, axis=0) + gbmid)
        gl = gz1 if HAS_MID else gz0
        Gm = tl.sum(gl[:, None, None] * gwout, axis=0) + gbout  # [PN,PM]
        Glin = tl.sum(a[:, None, None] * glw, axis=0) + glb  # [PN,PM]
        G = (Gm + Glin) * g_scale
        return G, dhdx, drift, hz0, hz1, jz0, jz1, rz0, rz1, gz0, gz1, sval

    @triton.jit
    def store_stage(zh0_p, zh1_p, zj0_p, zj1_p, zr0_p, zr1_p, zg0_p, zg1_p, sv_p,
                    off_z, off_s, rh,
                    hz0, hz1, jz0, jz1, rz0, rz1, gz0, gz1, sval,
                    HAS_BOUND: tl.constexpr, HAS_MID: tl.constexpr):
        tl.store(zh0_p + off_z + rh, hz0)
        tl.store(zj0_p + off_z + rh, jz0)
        tl.store(zr0_p + off_z + rh, rz0)
        tl.store(zg0_p + off_z + rh, gz0)
        if HAS_MID:
            tl.store(zh1_p + off_z + rh, hz1)
            tl.store(zj1_p + off_z + rh, jz1)
            tl.store(zr1_p + off_z + rh, rz1)
            tl.store(zg1_p + off_z + rh, gz1)
        if HAS_BOUND:
            tl.store(sv_p + off_s, sval)

    @triton.jit
    def phnn_fwd(u_ptr, x0_ptr, out_ptr, xs_ptr, qs_ptr,
                 zh0_p, zh1_p, zj0_p, zj1_p, zr0_p, zr1_p, zg0_p, zg1_p, sv_p,
                 hwin_p, hbin_p, hwmid_p, hbmid_p, hwout_p, hbout,
                 jwin_p, jbin_p, jwmid_p, jbmid_p, jwout_p, jbout_p,
                 rwin_p, rbin_p, rwmid_p, rbmid_p, rwout_p, rbout_p,
                 gwin_p, gbin_p, gwmid_p, gbmid_p, gwout_p, gbout_p, glw_p, glb_p, ow_p, ob_p,
                 jr_scale, g_scale, dt, bound, L,
                 STORE: tl.constexpr, HAS_BOUND: tl.constexpr, HAS_MID: tl.constexpr, OUT_LINEAR: tl.constexpr,
                 N: tl.constexpr, M: tl.constexpr, NY: tl.constexpr,
                 PN: tl.constexpr, PM: tl.constexpr, PNY: tl.constexpr, PH: tl.constexpr):
        pid = tl.program_id(0)
        rn = tl.arange(0, PN)
        rm = tl.arange(0, PM)
        rny = tl.arange(0, PNY)
        rh = tl.arange(0, PH)
        tiles = load_tiles(hwin_p, hbin_p, hwmid_p, hbmid_p, hwout_p, jwin_p, jbin_p, jwmid_p, jbmid_p, jwout_p,
                           jbout_p, rwin_p, rbin_p, rwmid_p, rbmid_p, rwout_p, rbout_p, gwin_p, gbin_p, gwmid_p,
                           gbmid_p, gwout_p, gbout_p, glw_p, glb_p, PN, PM, PH)
        ow = tl.load(ow_p + rn[:, None] * PNY + rny[None, :])
        ob = tl.load(ob_p + rny)
        x = tl.load(x0_ptr + pid * N + rn, mask=rn < N, other=0.0)  # [PN]
        for t in range(0, L):
            uu = tl.load(u_ptr + pid * (L * M) + t * M + rm, mask=rm < M, other=0.0)  # [PM]
            tl.store(xs_ptr + pid * (L * N) + t * N + rn, x, mask=rn < N)
            G, dhdx, drift, hz0, hz1, jz0, jz1, rz0, rz1, gz0, gz1, sval = fields(
                x, tiles, hbout, jr_scale, g_scale, bound, HAS_BOUND, HAS_MID, PN, PM, PH)
            if STORE:
                store_stage(zh0_p, zh1_p, zj0_p, zj1_p, zr0_p, zr1_p, zg0_p, zg1_p, sv_p,
                            pid * (L * 4 * PH) + t * (4 * PH) + 0 * PH, pid * (L * 4) + t * 4 + 0, rh,
                            hz0, hz1, jz0, jz1, rz0, rz1, gz0, gz1, sval, HAS_BOUND, HAS_MID)
            if OUT_LINEAR:
                y = tl.sum(x[:, None] * ow, axis=0) + ob  # [PNY]
                tl.store(out_ptr + pid * (L * NY) + t * NY + rny, y, mask=rny < NY)
            else:
                y = tl.sum(G * dhdx[:, None], axis=0)  # [PM] = G^T dhdx
                tl.store(out_ptr + pid * (L * NY) + t * NY + rm, y, mask=rm < NY)
            k1 = dt * (drift + tl.sum(G * uu[None, :], axis=1))
            q2 = x + k1 / 2
            G2, _, drift2, hz0, hz1, jz0, jz1, rz0, rz1, gz0, gz1, sval = fields(
                q2, tiles, hbout, jr_scale, g_scale, bound, HAS_BOUND, HAS_MID, PN, PM, PH)
            if STORE:
                tl.store(qs_ptr + pid * (L * 3 * PN) + t * (3 * PN) + 0 * PN + rn, q2)
                store_stage(zh0_p, zh1_p, zj0_p, zj1_p, zr0_p, zr1_p, zg0_p, zg1_p, sv_p,
                            pid * (L * 4 * PH) + t * (4 * PH) + 1 * PH, pid * (L * 4) + t * 4 + 1, rh,
                            hz0, hz1, jz0, jz1, rz0, rz1, gz0, gz1, sval, HAS_BOUND, HAS_MID)
            k2 = dt * (drift2 + tl.sum(G2 * uu[None, :], axis=1))
            q3 = x + k2 / 2
            G3, _, drift3, hz0, hz1, jz0, jz1, rz0, rz1, gz0, gz1, sval = fields(
                q3, tiles, hbout, jr_scale, g_scale, bound, HAS_BOUND, HAS_MID, PN, PM, PH)
            if STORE:
                tl.store(qs_ptr + pid * (L * 3 * PN) + t * (3 * PN) + 1 * PN + rn, q3)
                store_stage(zh0_p, zh1_p, zj0_p, zj1_p, zr0_p, zr1_p, zg0_p, zg1_p, sv_p,
                            pid * (L * 4 * PH) + t * (4 * PH) + 2 * PH, pid * (L * 4) + t * 4 + 2, rh,
                            hz0, hz1, jz0, jz1, rz0, rz1, gz0, gz1, sval, HAS_BOUND, HAS_MID)
            k3 = dt * (drift3 + tl.sum(G3 * uu[None, :], axis=1))
            q4 = x + k3
            G4, _, drift4, hz0, hz1, jz0, jz1, rz0, rz1, gz0, gz1, sval = fields(
                q4, tiles, hbout, jr_scale, g_scale, bound, HAS_BOUND, HAS_MID, PN, PM, PH)
            if STORE:
                tl.store(qs_ptr + pid * (L * 3 * PN) + t * (3 * PN) + 2 * PN + rn, q4)
                store_stage(zh0_p, zh1_p, zj0_p, zj1_p, zr0_p, zr1_p, zg0_p, zg1_p, sv_p,
                            pid * (L * 4 * PH) + t * (4 * PH) + 3 * PH, pid * (L * 4) + t * 4 + 3, rh,
                            hz0, hz1, jz0, jz1, rz0, rz1, gz0, gz1, sval, HAS_BOUND, HAS_MID)
            k4 = dt * (drift4 + tl.sum(G4 * uu[None, :], axis=1))
            x = x + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    _KERNELS = dict(tl=tl, triton=triton, fields=fields, _tanh=_tanh,
                    load_tiles=load_tiles, phnn_fwd=phnn_fwd)
    _build_bwd()
    return _KERNELS


def _build_bwd():
    import triton
    import triton.language as tl

    load_tiles = _KERNELS["load_tiles"]

    @triton.jit
    def stage_fields(a, z0h, z1h, z0j, z1j, z0r, z1r, z0g, z1g, sval, tiles,
                     jr_scale, g_scale, HAS_BOUND: tl.constexpr, HAS_MID: tl.constexpr,
                     PN: tl.constexpr, PM: tl.constexpr, PH: tl.constexpr):
        """Rebuild the stage quantities the reverse pass needs from STORED activations
        (no tanh evaluations): G, dhdx, dhdx_raw, mv, gz0 (H gradient-tape node), JR, A."""
        (hwin, hbin, hwmid, hbmid, hwout, jwin, jbin, jwmid, jbmid, jwout, jbout,
         rwin, rbin, rwmid, rbmid, rwout, rbout, gwin, gbin, gwmid, gbmid, gwout, gbout, glw, glb) = tiles
        zl = z1h if HAS_MID else z0h
        gp_last = hwout * (1.0 - zl * zl)
        if HAS_MID:
            gz0 = tl.sum(hwmid * gp_last[None, :], axis=1)
            gp0 = gz0 * (1.0 - z0h * z0h)
            dhdx_raw = tl.sum(hwin * gp0[None, :], axis=1)
        else:
            gz0 = hwout
            dhdx_raw = tl.sum(hwin * gp_last[None, :], axis=1)
        mv = 1.0
        if HAS_BOUND:
            mv = tl.where(sval > 0, 1.0, tl.exp(sval))
        dhdx = mv * dhdx_raw
        jl = z1j if HAS_MID else z0j
        B = (tl.sum(jl[:, None, None] * jwout, axis=0) + jbout) * jr_scale
        rl = z1r if HAS_MID else z0r
        A = (tl.sum(rl[:, None, None] * rwout, axis=0) + rbout) * jr_scale
        JR = (B - tl.trans(B)) - tl.sum(A[:, None, :] * A[None, :, :], axis=2)
        gl = z1g if HAS_MID else z0g
        Gm = tl.sum(gl[:, None, None] * gwout, axis=0) + gbout
        Glin = tl.sum(a[:, None, None] * glw, axis=0) + glb
        G = (Gm + Glin) * g_scale
        return G, dhdx, dhdx_raw, mv, gz0, gp_last, JR, A

    @triton.jit
    def mlp_bq(z0, z1, win, wmid, wout, bout,
               HAS_MID: tl.constexpr, PN: tl.constexpr, PH: tl.constexpr, POUT: tl.constexpr):
        """Input adjoint of a plain matrix-output MLP given the output adjoint (no accs)."""
        bzl = tl.sum(tl.sum(wout * bout[None, :, :], axis=2), axis=1)  # [PH]
        if HAS_MID:
            bp1 = bzl * (1.0 - z1 * z1)
            bz0 = tl.sum(wmid * bp1[None, :], axis=1)
            bp0 = bz0 * (1.0 - z0 * z0)
        else:
            bp0 = bzl * (1.0 - z0 * z0)
        return tl.sum(win * bp0[None, :], axis=1)  # [PN]

    @triton.jit
    def hnet_bq(z0, z1, win, wmid, wout, gz0, b_dhdx_raw, b_Hraw,
                HAS_MID: tl.constexpr, PN: tl.constexpr, PH: tl.constexpr):
        """Input adjoint of the Hamiltonian two-tape VJP (MATH.md section 3.3, no accs)."""
        bgp0 = tl.sum(win * b_dhdx_raw[:, None], axis=0)  # [PH]
        bgz0 = bgp0 * (1.0 - z0 * z0)
        bd0 = bgp0 * (gz0 if HAS_MID else wout)
        if HAS_MID:
            bgp1 = tl.sum(wmid * bgz0[:, None], axis=0)
            bd1 = bgp1 * wout
            bz1 = wout * b_Hraw + (-2.0 * z1 * bd1)
            bp1 = bz1 * (1.0 - z1 * z1)
            bz0 = tl.sum(wmid * bp1[None, :], axis=1) + (-2.0 * z0 * bd0)
        else:
            bz0 = wout * b_Hraw + (-2.0 * z0 * bd0)
        bp0 = bz0 * (1.0 - z0 * z0)
        return tl.sum(win * bp0[None, :], axis=1)  # [PN]

    @triton.jit
    def phnn_bwd(u_ptr, xs_ptr, qs_ptr, go_ptr, du_ptr, gx0_ptr,
                 zh0_p, zh1_p, zj0_p, zj1_p, zr0_p, zr1_p, zg0_p, zg1_p, sv_p,
                 bj_p, br_p, bg_p, bhd_p, bhr_p,
                 hwin_p, hbin_p, hwmid_p, hbmid_p, hwout_p, hbout,
                 jwin_p, jbin_p, jwmid_p, jbmid_p, jwout_p, jbout_p,
                 rwin_p, rbin_p, rwmid_p, rbmid_p, rwout_p, rbout_p,
                 gwin_p, gbin_p, gwmid_p, gbmid_p, gwout_p, gbout_p, glw_p, glb_p, ow_p,
                 jr_scale, g_scale, dt, L,
                 HAS_BOUND: tl.constexpr, HAS_MID: tl.constexpr, OUT_LINEAR: tl.constexpr,
                 N: tl.constexpr, M: tl.constexpr, NY: tl.constexpr,
                 PN: tl.constexpr, PM: tl.constexpr, PNY: tl.constexpr, PH: tl.constexpr):
        pid = tl.program_id(0)
        rn = tl.arange(0, PN)
        rm = tl.arange(0, PM)
        rny = tl.arange(0, PNY)
        rh = tl.arange(0, PH)
        mnn = rn[:, None] * PN + rn[None, :]
        mnm = rn[:, None] * PM + rm[None, :]
        tiles = load_tiles(hwin_p, hbin_p, hwmid_p, hbmid_p, hwout_p, jwin_p, jbin_p, jwmid_p, jbmid_p, jwout_p,
                           jbout_p, rwin_p, rbin_p, rwmid_p, rbmid_p, rwout_p, rbout_p, gwin_p, gbin_p, gwmid_p,
                           gbmid_p, gwout_p, gbout_p, glw_p, glb_p, PN, PM, PH)
        (hwin, hbin, hwmid, hbmid, hwout, jwin, jbin, jwmid, jbmid, jwout, jbout,
         rwin, rbin, rwmid, rbmid, rwout, rbout, gwin, gbin, gwmid, gbmid, gwout, gbout, glw, glb) = tiles
        ow = tl.load(ow_p + rn[:, None] * PNY + rny[None, :])

        bx = tl.zeros([PN], dtype=tl.float32)
        for ti in range(0, L):
            t = L - 1 - ti
            xt = tl.load(xs_ptr + pid * (L * N) + t * N + rn, mask=rn < N, other=0.0)
            q2 = tl.load(qs_ptr + pid * (L * 3 * PN) + t * (3 * PN) + 0 * PN + rn)
            q3 = tl.load(qs_ptr + pid * (L * 3 * PN) + t * (3 * PN) + 1 * PN + rn)
            q4 = tl.load(qs_ptr + pid * (L * 3 * PN) + t * (3 * PN) + 2 * PN + rn)
            uu = tl.load(u_ptr + pid * (L * M) + t * M + rm, mask=rm < M, other=0.0)
            gy = tl.load(go_ptr + pid * (L * NY) + t * NY + rny, mask=rny < NY, other=0.0)  # [PNY]

            bxin = bx
            bk1 = bx / 6
            bk2 = bx / 3
            bk3 = bx / 3
            bk4 = bx / 6
            du_t = tl.zeros([PM], dtype=tl.float32)

            # stages 4 -> 2 (no output adjoints), then stage 1 with them
            for s in tl.static_range(3, -1, -1):
                off_z = pid * (L * 4 * PH) + t * (4 * PH) + s * PH
                z0h = tl.load(zh0_p + off_z + rh)
                z0j = tl.load(zj0_p + off_z + rh)
                z0r = tl.load(zr0_p + off_z + rh)
                z0g = tl.load(zg0_p + off_z + rh)
                z1h = z0h
                z1j = z0j
                z1r = z0r
                z1g = z0g
                if HAS_MID:
                    z1h = tl.load(zh1_p + off_z + rh)
                    z1j = tl.load(zj1_p + off_z + rh)
                    z1r = tl.load(zr1_p + off_z + rh)
                    z1g = tl.load(zg1_p + off_z + rh)
                sval = 0.0
                if HAS_BOUND:
                    sval = tl.load(sv_p + pid * (L * 4) + t * 4 + s)
                a = xt
                if s == 1:
                    a = q2
                if s == 2:
                    a = q3
                if s == 3:
                    a = q4
                G, dhdx, dhdx_raw, mv, gz0, gp_last, JR, A = stage_fields(
                    a, z0h, z1h, z0j, z1j, z0r, z1r, z0g, z1g, sval, tiles,
                    jr_scale, g_scale, HAS_BOUND, HAS_MID, PN, PM, PH)
                brs = bk4
                if s == 2:
                    brs = bk3
                if s == 1:
                    brs = bk2
                if s == 0:
                    brs = bk1
                brs = dt * brs
                du_t += tl.sum(G * brs[:, None], axis=0)  # G^T brs
                bG = brs[:, None] * uu[None, :]
                bdhdx_ext = tl.zeros([PN], dtype=tl.float32)
                if s == 0:
                    if OUT_LINEAR:
                        bxin += tl.sum(ow * gy[None, :], axis=1)
                    else:
                        bG += dhdx[:, None] * gy[None, :]  # y = G^T dhdx (gy lives in rm lanes)
                        bdhdx_ext += tl.sum(G * gy[None, :], axis=1)
                # seeds (MATH.md section 3.2)
                bJR = brs[:, None] * dhdx[None, :]
                bdhdx = bdhdx_ext + tl.sum(JR * brs[:, None], axis=0)
                bjout = jr_scale * (bJR - tl.trans(bJR))
                bJRs = bJR + tl.trans(bJR)
                brout = jr_scale * (-tl.sum(bJRs[:, :, None] * A[None, :, :], axis=1))
                bgout = g_scale * bG
                b_dhdx_raw = mv * bdhdx
                b_Hraw = 0.0
                if HAS_BOUND:
                    b_Hraw = tl.sum(dhdx_raw * bdhdx) * tl.where(sval > 0, 0.0, tl.exp(sval))
                off_nn = pid * (L * 4 * PN * PN) + t * (4 * PN * PN) + s * (PN * PN)
                off_nm = pid * (L * 4 * PN * PM) + t * (4 * PN * PM) + s * (PN * PM)
                off_n = pid * (L * 4 * PN) + t * (4 * PN) + s * PN
                tl.store(bj_p + off_nn + mnn, bjout)
                tl.store(br_p + off_nn + mnn, brout)
                tl.store(bg_p + off_nm + mnm, bgout)
                tl.store(bhd_p + off_n + rn, b_dhdx_raw)
                tl.store(bhr_p + pid * (L * 4) + t * 4 + s, b_Hraw)
                # input adjoint of the stage
                bq = hnet_bq(z0h, z1h, hwin, hwmid, hwout, gz0, b_dhdx_raw, b_Hraw, HAS_MID, PN, PH)
                bq += mlp_bq(z0j, z1j, jwin, jwmid, jwout, bjout, HAS_MID, PN, PH, PN)
                bq += mlp_bq(z0r, z1r, rwin, rwmid, rwout, brout, HAS_MID, PN, PH, PN)
                bq += mlp_bq(z0g, z1g, gwin, gwmid, gwout, bgout, HAS_MID, PN, PH, PM)
                bq += tl.sum(tl.sum(glw * bgout[None, :, :], axis=2), axis=1)  # Wlin^T bgout
                bxin += bq
                if s == 3:
                    bk3 += bq
                if s == 2:
                    bk2 += bq / 2
                if s == 1:
                    bk1 += bq / 2
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
    """Ordered weight tensors for the kernels; fill absent mid layers with dummies."""
    dev = d["h_win"].device
    dummy_v = torch.zeros(ph, device=dev)
    dummy_m = torch.zeros(ph, ph, device=dev)

    def get(k):
        if k in d:
            return d[k]
        return dummy_m if k.endswith("wmid") else dummy_v

    return [get(k) for k in _ARG_KEYS]


def _num_warps(spec):
    # Weight tiles are the register-pressure driver; the reverse kernel carries no
    # gradient tiles anymore, so one heuristic serves both kernels (re-tuned on a 4090).
    return 2 if spec.hidden <= 64 else 4


def _zeros(*shape, dev):
    return torch.zeros(*shape, device=dev, dtype=torch.float32)


def _run_fwd(core, spec, u, x0, store: bool, num_warps: int | None = None):
    K = _build_kernels()
    d, (pn, pm, pny, ph, mid) = _prep(core, spec)
    sc = _scalars(core, spec)
    B, L = u.shape[0], u.shape[1]
    dev = u.device
    out = torch.empty(B, L, spec.n_output, device=dev, dtype=torch.float32)
    xs = torch.empty(B, L, spec.n_state, device=dev, dtype=torch.float32)
    dummy = torch.zeros(1, device=dev)
    if store:
        qs = torch.empty(B, L, 3, pn, device=dev, dtype=torch.float32)
        z0 = [torch.empty(B, L, 4, ph, device=dev, dtype=torch.float32) for _ in range(4)]
        z1 = [torch.empty(B, L, 4, ph, device=dev, dtype=torch.float32) for _ in range(4)] if mid else [dummy] * 4
        sv = torch.empty(B, L, 4, device=dev, dtype=torch.float32) if sc["has_bound"] else dummy
    else:
        qs = dummy
        z0 = [dummy] * 4
        z1 = [dummy] * 4
        sv = dummy
    wargs = _weight_args(d, ph)
    hbout = float(d["h_bout"])
    ow = d.get("o_w", torch.zeros(pn, pny, device=dev))
    ob = d.get("o_b", torch.zeros(pny, device=dev))
    K["phnn_fwd"][(B,)](
        u.contiguous(), x0.contiguous(), out, xs, qs,
        z0[0], z1[0], z0[1], z1[1], z0[2], z1[2], z0[3], z1[3], sv,
        wargs[0], wargs[1], wargs[2], wargs[3], wargs[4], hbout, *wargs[5:], ow, ob,
        sc["jr_scale"], sc["g_scale"], sc["dt"], sc["bound"], L,
        store, sc["has_bound"], mid, sc["out_linear"],
        sc["n"], sc["m"], sc["ny"], pn, pm, pny, ph,
        num_warps=_num_warps(spec) if num_warps is None else num_warps,
    )
    saved = (xs, qs, *z0, *z1, sv) if store else None
    return out, saved


def _run_bwd(core, spec, u, saved, grad_out, num_warps: int | None = None):
    K = _build_kernels()
    d, (pn, pm, pny, ph, mid) = _prep(core, spec)
    sc = _scalars(core, spec)
    xs, qs, zh0, zj0, zr0, zg0, zh1, zj1, zr1, zg1, sv = saved
    B, L = xs.shape[0], xs.shape[1]
    dev = u.device
    wargs = _weight_args(d, ph)
    hbout = float(d["h_bout"])
    ow = d.get("o_w", _zeros(pn, pny, dev=dev))
    du = _zeros(B, L, spec.n_input, dev=dev)
    gx0 = _zeros(B, spec.n_state, dev=dev)
    bj = torch.empty(B, L, 4, pn, pn, device=dev, dtype=torch.float32)
    br = torch.empty(B, L, 4, pn, pn, device=dev, dtype=torch.float32)
    bg = torch.empty(B, L, 4, pn, pm, device=dev, dtype=torch.float32)
    bhd = torch.empty(B, L, 4, pn, device=dev, dtype=torch.float32)
    bhr = torch.empty(B, L, 4, device=dev, dtype=torch.float32)
    K["phnn_bwd"][(B,)](
        u.contiguous(), xs, qs, grad_out.contiguous(), du, gx0,
        zh0, zh1, zj0, zj1, zr0, zr1, zg0, zg1, sv,
        bj, br, bg, bhd, bhr,
        wargs[0], wargs[1], wargs[2], wargs[3], wargs[4], hbout, *wargs[5:], ow,
        sc["jr_scale"], sc["g_scale"], sc["dt"], L,
        sc["has_bound"], mid, sc["out_linear"],
        sc["n"], sc["m"], sc["ny"], pn, pm, pny, ph,
        num_warps=_num_warps(spec) if num_warps is None else num_warps,
    )
    grads = _param_grads(core, spec, saved, grad_out, (bj, br, bg, bhd, bhr))
    return grads, du, gx0


def _param_grads(core, spec, saved, grad_out, seeds):
    """Parameter gradients as batched GEMMs over the ``[B*L*4, .]`` flattened per-stage
    adjoint seeds emitted by ``phnn_bwd`` (MATH.md sections 3.3/3.4)."""
    p = params_of(core)
    n, m, ny, h = spec.n_state, spec.n_input, spec.n_output, spec.hidden
    mid = spec.num_layers == 2
    xs, qs, zh0, zj0, zr0, zg0, zh1, zj1, zr1, zg1, sv = saved
    bj, br, bg, bhd, bhr = seeds
    B, L = xs.shape[0], xs.shape[1]
    F = B * L * 4
    # stage input states a: [B, L, 4, n] = (x_t, q2, q3, q4)
    a = torch.cat((xs.unsqueeze(2), qs[..., :n]), dim=2).reshape(F, n)

    def flat_z(z):
        return z[..., :h].reshape(F, h)

    def plain(bout, z0, z1, w):
        """Grads of a plain MLP with matrix output layer; ``bout`` [F, o] is the raw
        output adjoint, ``w`` the [w0, (w1,), wout] list of live weights."""
        zl = z1 if mid else z0
        gw_out = bout.t() @ zl
        gb_out = bout.sum(0)
        bzl = bout @ w[-1]
        if mid:
            bp1 = bzl * (1.0 - z1 * z1)
            gw1 = bp1.t() @ z0
            gb1 = bp1.sum(0)
            bz0 = bp1 @ w[1]
            bp0 = bz0 * (1.0 - z0 * z0)
        else:
            bp0 = bzl * (1.0 - z0 * z0)
        gw0 = bp0.t() @ a
        gb0 = bp0.sum(0)
        gws = [gw0, gw1, gw_out] if mid else [gw0, gw_out]
        gbs = [gb0, gb1, gb_out] if mid else [gb0, gb_out]
        return gws, gbs

    with torch.no_grad():
        # ---- Hamiltonian (two coupled tapes, section 3.3) ----
        hw = [w.detach() for w in p["hw"]]
        z0 = flat_z(zh0)
        z1 = flat_z(zh1) if mid else z0
        w_out = hw[-1].reshape(h)
        b_dhdx_raw = bhd[..., :n].reshape(F, n)
        b_Hraw = bhr.reshape(F, 1)
        zl = z1 if mid else z0
        gp_last = (1.0 - zl * zl) * w_out
        bgp0 = b_dhdx_raw @ hw[0].t()  # [F,h]
        bgz0 = bgp0 * (1.0 - z0 * z0)
        if mid:
            gz0 = gp_last @ hw[1]
            gp0 = gz0 * (1.0 - z0 * z0)
            hW0 = gp0.t() @ b_dhdx_raw  # gradient tape
            bd0 = bgp0 * gz0
            hW1 = gp_last.t() @ bgz0
            bgp1 = bgz0 @ hw[1].t()
            bgz1 = bgp1 * (1.0 - z1 * z1)
            bd1 = bgp1 * w_out
            hWout = bgz1.sum(0)
            # value tape
            bz1 = w_out * b_Hraw - 2.0 * z1 * bd1
            hWout = hWout + (b_Hraw * z1).sum(0)
            hbout = b_Hraw.sum()
            bp1 = bz1 * (1.0 - z1 * z1)
            hW1 = hW1 + bp1.t() @ z0
            hb1 = bp1.sum(0)
            bz0 = bp1 @ hw[1] - 2.0 * z0 * bd0
        else:
            hW0 = gp_last.t() @ b_dhdx_raw
            bd0 = bgp0 * w_out
            hWout = bgz0.sum(0)
            bz0 = w_out * b_Hraw - 2.0 * z0 * bd0
            hWout = hWout + (b_Hraw * z0).sum(0)
            hbout = b_Hraw.sum()
        bp0 = bz0 * (1.0 - z0 * z0)
        hW0 = hW0 + bp0.t() @ a
        hb0 = bp0.sum(0)
        h_gw = [hW0, hW1, hWout.reshape(1, h)] if mid else [hW0, hWout.reshape(1, h)]
        h_gb = [hb0, hb1, hbout.reshape(1)] if mid else [hb0, hbout.reshape(1)]

        # ---- J, R, G nets (section 3.4) ----
        bj_f = bj[..., :n, :n].reshape(F, n * n)
        br_f = br[..., :n, :n].reshape(F, n * n)
        bg_f = bg[..., :n, :m].reshape(F, n * m)
        j_gw, j_gb = plain(bj_f, flat_z(zj0), flat_z(zj1) if mid else None, [w.detach() for w in p["jw"]])
        r_gw, r_gb = plain(br_f, flat_z(zr0), flat_z(zr1) if mid else None, [w.detach() for w in p["rw"]])
        g_gw, g_gb = plain(bg_f, flat_z(zg0), flat_z(zg1) if mid else None, [w.detach() for w in p["gw"]])
        # G linear bypass
        gl_w = bg_f.t() @ a
        gl_b = bg_f.sum(0)

        grads = [*h_gw, *h_gb, *j_gw, *j_gb, *r_gw, *r_gb, gl_w, gl_b, *g_gw, *g_gb]
        if spec.output == "linear":
            gy = grad_out.reshape(B * L, ny)
            grads += [gy.t() @ xs.reshape(B * L, n), gy.sum(0)]
    return [g.contiguous() for g in grads]


class _TritonPHNNRollout(torch.autograd.Function):
    @staticmethod
    def forward(ctx, core, spec, u, x0, *params):
        out, saved = _run_fwd(core, spec, u.contiguous(), x0.contiguous(), store=True)
        ctx.core, ctx.spec = core, spec
        ctx.save_for_backward(u, *saved)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        u, *saved = ctx.saved_tensors
        grads, du, gx0 = _run_bwd(ctx.core, ctx.spec, u, saved, grad_out.contiguous())
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
        out, _ = _run_fwd(core, spec, u.contiguous(), x0.contiguous(), store=False)
        return out
    return _TritonPHNNRollout.apply(core, spec, u.contiguous(), x0.contiguous(), *params)
