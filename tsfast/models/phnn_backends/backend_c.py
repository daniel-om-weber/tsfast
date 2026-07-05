"""Generated-C++ execution backend for the PHNN core: fused CPU rollout and BPTT.

Unlike the SSM/NARX C backends (which emit spec-specialized C++ per layer spec), the
PHNN step is far more intricate — four component nets, a closed-form Hamiltonian
state-gradient, and a 4-stage RK4 update — so this backend compiles ONE generic,
scalar-templated kernel (``float`` and ``double``) with the dimensions passed at
runtime. Being fp64-capable it is the ``torch.autograd.gradcheck`` vehicle; it is also
the fast CPU path. The batch lanes run in parallel (ATen thread pool / OpenMP); each
lane owns a private slice of the per-lane gradient buffers, summed to parameter
gradients in Python (no locks, no races).

The forward stores only the per-step input states (B×L×n); the backward recomputes
each step's intra-step activations and runs the hand-derived reverse pass in
``MATH.md`` (§3), including the second-order coupling through the Hamiltonian gradient.
"""

__all__ = [
    "c_rollout",
    "is_available",
]

import hashlib

import torch

from ..ssm.backend_c import _build_flags, is_available  # toolchain probe + flags shared
from .common import PHNNSpec, bound_value, params_of

_EXTENSION = None

_SRC = r"""
#include <torch/extension.h>
#include <pybind11/stl.h>
#include <ATen/Parallel.h>
#include <vector>
#include <cmath>

constexpr int MAXH = 512;    // max hidden width
constexpr int MAXHID = 4;    // max hidden layers (K-1)
constexpr int MAXNN = 1152;  // max n_state*n_state (n_state <= 32) and other n-sized vectors

template <typename S>
struct Params {
  const S* const* hw; const S* const* hb; const S* const* jw; const S* const* jb;
  const S* const* rw; const S* const* rb;
  const S* glw; const S* glb; const S* const* gw; const S* const* gb; const S* ow; const S* ob;
  int n, nu, ny, nh, K, out_linear, has_bound;
  S dt, bound, jr_scale, g_scale;
};

template <typename S>
struct Grads {  // pointers already offset to the current lane
  S** hw; S** hb; S** jw; S** jb; S** rw; S** rb;
  S* glw; S* glb; S** gw; S** gb; S* ow; S* ob;
};

template <typename S>
struct StageBuf {
  S* Hz; S* Hgz; S* Hgp; S* dhdx_raw; S* Hraw; S* sval; S* mval;
  S* Jz; S* Rz; S* Gz; S* Bm; S* Am; S* JRm; S* Gm; S* dhdx; S* drift; S* q;
};

// ------------------------------------------------------------------ forward pieces
// Plain MLP forward (tanh hidden, linear output); stores hidden post-activations z[l].
template <typename S>
void mlp_fwd(const S* const* W, const S* const* B, int K, int n_in, int nh, int n_out,
             const S* a, S* z, S* out) {
  const S* prev = a; int pd = n_in;
  for (int l = 0; l < K; l++) {
    int no = (l == K - 1) ? n_out : nh;
    S* dst = (l == K - 1) ? out : (z + l * nh);
    for (int o = 0; o < no; o++) {
      const S* wr = W[l] + o * pd; S acc = B[l][o];
      for (int j = 0; j < pd; j++) acc += wr[j] * prev[j];
      dst[o] = acc;
    }
    if (l < K - 1) for (int o = 0; o < no; o++) dst[o] = std::tanh(dst[o]);
    prev = dst; pd = no;
  }
}

// Hamiltonian net: value forward + closed-form gradient tape (MATH.md §1.1).
template <typename S>
void hnet_fwd(const S* const* W, const S* const* B, int K, int n, int nh,
              const S* a, S* z, S* gz, S* gp, S* Hraw, S* dhdx_raw) {
  const S* prev = a; int pd = n;
  for (int l = 0; l < K; l++) {
    int no = (l == K - 1) ? 1 : nh;
    S* dst = (l == K - 1) ? Hraw : (z + l * nh);
    for (int o = 0; o < no; o++) {
      const S* wr = W[l] + o * pd; S acc = B[l][o];
      for (int j = 0; j < pd; j++) acc += wr[j] * prev[j];
      dst[o] = acc;
    }
    if (l < K - 1) for (int o = 0; o < no; o++) dst[o] = std::tanh(dst[o]);
    prev = dst; pd = no;
  }
  for (int k = 0; k < nh; k++) gz[(K - 2) * nh + k] = W[K - 1][k];  // gz_{K-2} = W_{K-1}^T . 1
  for (int l = K - 2; l >= 0; l--) {
    for (int k = 0; k < nh; k++) { S zl = z[l * nh + k]; gp[l * nh + k] = gz[l * nh + k] * (S(1) - zl * zl); }
    if (l > 0) {
      for (int j = 0; j < nh; j++) { S acc = 0; for (int o = 0; o < nh; o++) acc += W[l][o * nh + j] * gp[l * nh + o]; gz[(l - 1) * nh + j] = acc; }
    } else {
      for (int j = 0; j < n; j++) { S acc = 0; for (int o = 0; o < nh; o++) acc += W[0][o * n + j] * gp[o]; dhdx_raw[j] = acc; }
    }
  }
}

template <typename S>
void fields_fwd(const Params<S>& P, const S* a, StageBuf<S>& s) {
  const int n = P.n, nu = P.nu, nh = P.nh, K = P.K;
  hnet_fwd(P.hw, P.hb, K, n, nh, a, s.Hz, s.Hgz, s.Hgp, s.Hraw, s.dhdx_raw);
  S sv = 0, mv = 1;
  if (P.has_bound) { sv = *s.Hraw - P.bound; mv = (sv > 0) ? S(1) : std::exp(sv); }
  *s.sval = sv; *s.mval = mv;
  for (int i = 0; i < n; i++) s.dhdx[i] = mv * s.dhdx_raw[i];
  mlp_fwd(P.jw, P.jb, K, n, nh, n * n, a, s.Jz, s.Bm);
  for (int i = 0; i < n * n; i++) s.Bm[i] *= P.jr_scale;
  mlp_fwd(P.rw, P.rb, K, n, nh, n * n, a, s.Rz, s.Am);
  for (int i = 0; i < n * n; i++) s.Am[i] *= P.jr_scale;
  mlp_fwd(P.gw, P.gb, K, n, nh, n * nu, a, s.Gz, s.Gm);
  for (int i = 0; i < n * nu; i++) {
    S lin = P.glb[i]; const S* wr = P.glw + i * n; for (int j = 0; j < n; j++) lin += wr[j] * a[j];
    s.Gm[i] = (s.Gm[i] + lin) * P.g_scale;
  }
  for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) {
    S r = 0; for (int k = 0; k < n; k++) r += s.Am[i * n + k] * s.Am[j * n + k];
    s.JRm[i * n + j] = (s.Bm[i * n + j] - s.Bm[j * n + i]) - r;
  }
  for (int i = 0; i < n; i++) { S d = 0; for (int j = 0; j < n; j++) d += s.JRm[i * n + j] * s.dhdx[j]; s.drift[i] = d; }
}

// ------------------------------------------------------------------ backward pieces
// Plain MLP VJP (linear output dim n_out); accumulates dW,dB and adds input adjoint to bq.
template <typename S>
void mlp_vjp(const S* const* W, int K, int n_in, int nh, int n_out,
             const S* z, const S* a, const S* bout, S** dW, S** dB, S* bq) {
  S cur[MAXH];
  { const S* zin = z + (K - 2) * nh;
    for (int j = 0; j < nh; j++) { S acc = 0; for (int o = 0; o < n_out; o++) acc += W[K - 1][o * nh + j] * bout[o]; cur[j] = acc; }
    for (int o = 0; o < n_out; o++) { dB[K - 1][o] += bout[o]; S* wr = dW[K - 1] + o * nh; for (int j = 0; j < nh; j++) wr[j] += bout[o] * zin[j]; } }
  for (int l = K - 2; l >= 0; l--) {
    int in_dim = (l == 0) ? n_in : nh; const S* inp = (l == 0) ? a : (z + (l - 1) * nh);
    const S* zl = z + l * nh; S bp[MAXH];
    for (int o = 0; o < nh; o++) { S zv = zl[o]; bp[o] = cur[o] * (S(1) - zv * zv); }
    for (int o = 0; o < nh; o++) { dB[l][o] += bp[o]; S* wr = dW[l] + o * in_dim; for (int j = 0; j < in_dim; j++) wr[j] += bp[o] * inp[j]; }
    if (l > 0) for (int j = 0; j < in_dim; j++) { S acc = 0; for (int o = 0; o < nh; o++) acc += W[l][o * in_dim + j] * bp[o]; cur[j] = acc; }
    else for (int j = 0; j < in_dim; j++) { S acc = 0; for (int o = 0; o < nh; o++) acc += W[0][o * in_dim + j] * bp[o]; bq[j] += acc; }
  }
}

// Hamiltonian VJP: gradient tape + value tape sharing W_0..W_{K-1} (MATH.md §3.3).
template <typename S>
void hnet_vjp(const S* const* W, int K, int n, int nh, const S* z, const S* gz, const S* gp,
              const S* a, const S* b_dhdx_raw, S b_Hraw, S** dW, S** dB, S* bq) {
  S bgz[MAXH], bgp[MAXH], bd[MAXHID * MAXH];
  // gradient tape reverse
  for (int o = 0; o < nh; o++) { S* wr = dW[0] + o * n; S gpo = gp[o]; for (int j = 0; j < n; j++) wr[j] += b_dhdx_raw[j] * gpo; }
  for (int o = 0; o < nh; o++) { S acc = 0; for (int j = 0; j < n; j++) acc += W[0][o * n + j] * b_dhdx_raw[j]; bgp[o] = acc; }
  for (int k = 0; k < nh; k++) { S zl = z[k]; S d0 = S(1) - zl * zl; bgz[k] = bgp[k] * d0; bd[k] = bgp[k] * gz[k]; }
  for (int l = 1; l <= K - 2; l++) {
    for (int o = 0; o < nh; o++) { S* wr = dW[l] + o * nh; S gpo = gp[l * nh + o]; for (int j = 0; j < nh; j++) wr[j] += bgz[j] * gpo; }
    for (int o = 0; o < nh; o++) { S acc = 0; for (int j = 0; j < nh; j++) acc += W[l][o * nh + j] * bgz[j]; bgp[o] = acc; }
    for (int k = 0; k < nh; k++) { S zl = z[l * nh + k]; S dl = S(1) - zl * zl; bgz[k] = bgp[k] * dl; bd[l * nh + k] = bgp[k] * gz[l * nh + k]; }
  }
  for (int k = 0; k < nh; k++) dW[K - 1][k] += bgz[k];  // gz_{K-2} = W_{K-1}^T . 1
  // value tape reverse
  S bz[MAXH];
  { const S* zt = z + (K - 2) * nh;
    for (int k = 0; k < nh; k++) { bz[k] = W[K - 1][k] * b_Hraw + (S(-2) * zt[k] * bd[(K - 2) * nh + k]); dW[K - 1][k] += b_Hraw * zt[k]; }
    dB[K - 1][0] += b_Hraw; }
  for (int l = K - 2; l >= 0; l--) {
    int in_dim = (l == 0) ? n : nh; const S* inp = (l == 0) ? a : (z + (l - 1) * nh);
    const S* zl = z + l * nh; S bp[MAXH];
    for (int o = 0; o < nh; o++) { S zv = zl[o]; bp[o] = bz[o] * (S(1) - zv * zv); }
    for (int o = 0; o < nh; o++) { dB[l][o] += bp[o]; S* wr = dW[l] + o * in_dim; for (int j = 0; j < in_dim; j++) wr[j] += bp[o] * inp[j]; }
    if (l > 0) for (int j = 0; j < in_dim; j++) { S acc = 0; for (int o = 0; o < nh; o++) acc += W[l][o * in_dim + j] * bp[o]; bz[j] = acc + (S(-2) * inp[j] * bd[(l - 1) * nh + j]); }
    else for (int j = 0; j < in_dim; j++) { S acc = 0; for (int o = 0; o < nh; o++) acc += W[0][o * in_dim + j] * bp[o]; bq[j] += acc; }
  }
}

// fields VJP (MATH.md §3.2); accumulates all component-net grads, returns input adjoint in bq.
template <typename S>
void fields_vjp(const Params<S>& P, Grads<S>& g, const StageBuf<S>& s,
                const S* bG_in, const S* bdhdx_ext, const S* bdrift, S* bq) {
  const int n = P.n, nu = P.nu, nh = P.nh, K = P.K;
  S bJR[MAXNN], bdhdx[MAXNN], bjout[MAXNN], brout[MAXNN], bgout[MAXNN], bdraw[MAXNN];
  for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) bJR[i * n + j] = bdrift[i] * s.dhdx[j];
  for (int j = 0; j < n; j++) { S acc = bdhdx_ext[j]; for (int i = 0; i < n; i++) acc += s.JRm[i * n + j] * bdrift[i]; bdhdx[j] = acc; }
  for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) bjout[i * n + j] = P.jr_scale * (bJR[i * n + j] - bJR[j * n + i]);
  for (int i = 0; i < n; i++) for (int k = 0; k < n; k++) {
    S acc = 0; for (int j = 0; j < n; j++) acc += -(bJR[i * n + j] + bJR[j * n + i]) * s.Am[j * n + k];
    brout[i * n + k] = P.jr_scale * acc;
  }
  for (int i = 0; i < n * nu; i++) bgout[i] = P.g_scale * bG_in[i];
  S b_Hraw = 0;
  S mv = *s.mval;
  for (int i = 0; i < n; i++) bdraw[i] = mv * bdhdx[i];
  if (P.has_bound) {
    S bm = 0; for (int i = 0; i < n; i++) bm += s.dhdx_raw[i] * bdhdx[i];
    S sv = *s.sval; b_Hraw = bm * ((sv > 0) ? S(0) : std::exp(sv));
  }
  hnet_vjp(P.hw, K, n, nh, s.Hz, s.Hgz, s.Hgp, s.q, bdraw, b_Hraw, g.hw, g.hb, bq);
  mlp_vjp(P.jw, K, n, nh, n * n, s.Jz, s.q, bjout, g.jw, g.jb, bq);
  mlp_vjp(P.rw, K, n, nh, n * n, s.Rz, s.q, brout, g.rw, g.rb, bq);
  // g-net ResMLP: linear bypass + mlp, both take bgout
  for (int o = 0; o < n * nu; o++) { g.glb[o] += bgout[o]; S* wr = g.glw + o * n; for (int j = 0; j < n; j++) wr[j] += bgout[o] * s.q[j]; }
  for (int j = 0; j < n; j++) { S acc = 0; for (int o = 0; o < n * nu; o++) acc += P.glw[o * n + j] * bgout[o]; bq[j] += acc; }
  mlp_vjp(P.gw, K, n, nh, n * nu, s.Gz, s.q, bgout, g.gw, g.gb, bq);
}

// ------------------------------------------------------------------ pointer gather helpers
template <typename S> std::vector<const S*> cptrs(const std::vector<torch::Tensor>& v) {
  std::vector<const S*> p; for (auto& t : v) p.push_back(t.data_ptr<S>()); return p;
}

template <typename S>
void fill_params(Params<S>& P,
    const std::vector<const S*>& hw, const std::vector<const S*>& hb,
    const std::vector<const S*>& jw, const std::vector<const S*>& jb,
    const std::vector<const S*>& rw, const std::vector<const S*>& rb,
    const S* glw, const S* glb, const std::vector<const S*>& gw, const std::vector<const S*>& gb,
    const S* ow, const S* ob) {
  P.hw = hw.data(); P.hb = hb.data(); P.jw = jw.data(); P.jb = jb.data();
  P.rw = rw.data(); P.rb = rb.data(); P.gw = gw.data(); P.gb = gb.data();
  P.glw = glw; P.glb = glb; P.ow = ow; P.ob = ob;
}

// ------------------------------------------------------------------ forward driver
template <typename S>
void fwd_impl(const Params<S>& P, const S* x0, const S* u, S* out, S* xstates, int64_t B, int64_t L) {
  const int n = P.n, nu = P.nu, ny = P.ny;
  at::parallel_for(0, B, 1, [&](int64_t b0, int64_t b1) {
    // single-stage scratch reused across all steps
    std::vector<S> Hz((P.K - 1) * P.nh), Hgz((P.K - 1) * P.nh), Hgp((P.K - 1) * P.nh), draw(n);
    std::vector<S> Jz((P.K - 1) * P.nh), Rz((P.K - 1) * P.nh), Gz((P.K - 1) * P.nh);
    std::vector<S> Bm(n * n), Am(n * n), JRm(n * n), Gm(n * nu), dhdx(n), drift(n);
    S Hraw, sval, mval;
    StageBuf<S> s{Hz.data(), Hgz.data(), Hgp.data(), draw.data(), &Hraw, &sval, &mval,
                  Jz.data(), Rz.data(), Gz.data(), Bm.data(), Am.data(), JRm.data(), Gm.data(), dhdx.data(), drift.data(), nullptr};
    std::vector<S> x(n), q(n), k1(n), k2(n), k3(n), k4(n), rhs(n);
    for (int64_t b = b0; b < b1; b++) {
      for (int i = 0; i < n; i++) x[i] = x0[b * n + i];
      for (int64_t t = 0; t < L; t++) {
        const S* ut = u + (b * L + t) * nu;
        S* xs = xstates + (b * L + t) * n; for (int i = 0; i < n; i++) xs[i] = x[i];
        s.q = x.data();
        fields_fwd(P, x.data(), s);
        S* yo = out + (b * L + t) * ny;
        if (P.out_linear) { for (int o = 0; o < ny; o++) { S acc = P.ob[o]; const S* wr = P.ow + o * n; for (int j = 0; j < n; j++) acc += wr[j] * x[j]; yo[o] = acc; } }
        else { for (int j = 0; j < ny; j++) { S acc = 0; for (int i = 0; i < n; i++) acc += Gm[i * nu + j] * dhdx[i]; yo[j] = acc; } }
        for (int i = 0; i < n; i++) { S gu = 0; for (int j = 0; j < nu; j++) gu += Gm[i * nu + j] * ut[j]; k1[i] = P.dt * (drift[i] + gu); }
        for (int i = 0; i < n; i++) q[i] = x[i] + k1[i] / 2; s.q = q.data(); fields_fwd(P, q.data(), s);
        for (int i = 0; i < n; i++) { S gu = 0; for (int j = 0; j < nu; j++) gu += Gm[i * nu + j] * ut[j]; k2[i] = P.dt * (drift[i] + gu); }
        for (int i = 0; i < n; i++) q[i] = x[i] + k2[i] / 2; s.q = q.data(); fields_fwd(P, q.data(), s);
        for (int i = 0; i < n; i++) { S gu = 0; for (int j = 0; j < nu; j++) gu += Gm[i * nu + j] * ut[j]; k3[i] = P.dt * (drift[i] + gu); }
        for (int i = 0; i < n; i++) q[i] = x[i] + k3[i]; s.q = q.data(); fields_fwd(P, q.data(), s);
        for (int i = 0; i < n; i++) { S gu = 0; for (int j = 0; j < nu; j++) gu += Gm[i * nu + j] * ut[j]; k4[i] = P.dt * (drift[i] + gu); }
        for (int i = 0; i < n; i++) x[i] = x[i] + (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]) / 6;
      }
    }
  });
}

// ------------------------------------------------------------------ backward driver
template <typename S>
void bwd_impl(const Params<S>& P, const S* u, const S* xstates, const S* grad_out,
              Grads<S>* lane, S* du, S* gx0, int64_t B, int64_t L, int64_t pf_stride,
              S* pf) {  // pf: [B, pf_stride] flat per-lane param grads (lane->pointers index into it)
  const int n = P.n, nu = P.nu, ny = P.ny, nh = P.nh, K = P.K;
  at::parallel_for(0, B, 1, [&](int64_t b0, int64_t b1) {
    // 4-stage bundles
    int hlen = (K - 1) * nh;
    std::vector<S> Hz(4 * hlen), Hgz(4 * hlen), Hgp(4 * hlen), draw(4 * n);
    std::vector<S> Jz(4 * hlen), Rz(4 * hlen), Gz(4 * hlen);
    std::vector<S> Bm(4 * n * n), Am(4 * n * n), JRm(4 * n * n), Gm(4 * n * nu), dhdx(4 * n), drift(4 * n);
    std::vector<S> Hraw(4), sval(4), mval(4), qbuf(4 * n), kbuf(4 * n);
    StageBuf<S> sb[4];
    for (int st = 0; st < 4; st++) {
      sb[st] = StageBuf<S>{Hz.data() + st * hlen, Hgz.data() + st * hlen, Hgp.data() + st * hlen, draw.data() + st * n,
        &Hraw[st], &sval[st], &mval[st], Jz.data() + st * hlen, Rz.data() + st * hlen, Gz.data() + st * hlen,
        Bm.data() + st * n * n, Am.data() + st * n * n, JRm.data() + st * n * n, Gm.data() + st * n * nu,
        dhdx.data() + st * n, drift.data() + st * n, qbuf.data() + st * n};
    }
    std::vector<S> bx(n), bxin(n), bk[4], brhs(n), bq(n), bG(MAXNN), bdhdx_ext(n), du_t(nu), zeros_n(n, 0), bGs(n * nu);
    for (int st = 0; st < 4; st++) bk[st].resize(n);
    for (int64_t b = b0; b < b1; b++) {
      Grads<S>& g = lane[b];
      for (int i = 0; i < n; i++) bx[i] = 0;
      for (int64_t ti = 0; ti < L; ti++) {
        int64_t t = L - 1 - ti;
        const S* xt = xstates + (b * L + t) * n;
        const S* ut = u + (b * L + t) * nu;
        const S* gy = grad_out + (b * L + t) * ny;
        // ---- recompute forward, storing the 4 stages ----
        for (int i = 0; i < n; i++) sb[0].q[i] = xt[i];
        fields_fwd(P, xt, sb[0]);
        auto stage_k = [&](int st, S* kout) { for (int i = 0; i < n; i++) { S gu = 0; for (int j = 0; j < nu; j++) gu += sb[st].Gm[i * nu + j] * ut[j]; kout[i] = P.dt * (sb[st].drift[i] + gu); } };
        S* K1 = kbuf.data(); S* K2 = kbuf.data() + n; S* K3 = kbuf.data() + 2 * n; S* K4 = kbuf.data() + 3 * n;
        stage_k(0, K1);
        for (int i = 0; i < n; i++) sb[1].q[i] = xt[i] + K1[i] / 2; fields_fwd(P, sb[1].q, sb[1]); stage_k(1, K2);
        for (int i = 0; i < n; i++) sb[2].q[i] = xt[i] + K2[i] / 2; fields_fwd(P, sb[2].q, sb[2]); stage_k(2, K3);
        for (int i = 0; i < n; i++) sb[3].q[i] = xt[i] + K3[i]; fields_fwd(P, sb[3].q, sb[3]); stage_k(3, K4);
        // ---- reverse ----
        for (int i = 0; i < n; i++) bxin[i] = bx[i];  // direct x_next -> x term (gxn carried)
        for (int i = 0; i < n; i++) { bk[0][i] = bx[i] / 6; bk[3][i] = bx[i] / 6; bk[1][i] = bx[i] / 3; bk[2][i] = bx[i] / 3; }
        for (int i = 0; i < MAXNN && i < n * nu; i++) bG[i] = 0;
        for (int i = 0; i < n; i++) bdhdx_ext[i] = 0;
        for (int j = 0; j < nu; j++) du_t[j] = 0;
        if (P.out_linear) {
          for (int i = 0; i < n; i++) { S acc = 0; for (int o = 0; o < ny; o++) acc += P.ow[o * n + i] * gy[o]; bxin[i] += acc; }
          for (int o = 0; o < ny; o++) { g.ob[o] += gy[o]; S* wr = g.ow + o * n; for (int i = 0; i < n; i++) wr[i] += gy[o] * xt[i]; }
        } else {
          for (int i = 0; i < n; i++) for (int j = 0; j < nu; j++) bG[i * nu + j] += gy[j] * sb[0].dhdx[i];
          for (int i = 0; i < n; i++) { S acc = 0; for (int j = 0; j < nu; j++) acc += sb[0].Gm[i * nu + j] * gy[j]; bdhdx_ext[i] += acc; }
        }
        // stages 3,2,1 (k4,k3,k2)
        for (int st = 3; st >= 1; st--) {
          for (int i = 0; i < n; i++) brhs[i] = P.dt * bk[st][i];
          for (int i = 0; i < n; i++) for (int j = 0; j < nu; j++) bGs[i * nu + j] = brhs[i] * ut[j];
          for (int j = 0; j < nu; j++) { S acc = 0; for (int i = 0; i < n; i++) acc += sb[st].Gm[i * nu + j] * brhs[i]; du_t[j] += acc; }
          for (int i = 0; i < n; i++) bq[i] = 0;
          // stages 1..3 have no external dhdx adjoint (dhdx used only inside drift)
          fields_vjp(P, g, sb[st], bGs.data(), zeros_n.data(), brhs.data(), bq.data());
          for (int i = 0; i < n; i++) bxin[i] += bq[i];
          if (st == 3) for (int i = 0; i < n; i++) bk[2][i] += bq[i];
          else if (st == 2) for (int i = 0; i < n; i++) bk[1][i] += bq[i] / 2;
          else for (int i = 0; i < n; i++) bk[0][i] += bq[i] / 2;
        }
        // stage 0 (k1) with output adjoints already in bG / bdhdx_ext
        for (int i = 0; i < n; i++) brhs[i] = P.dt * bk[0][i];
        for (int i = 0; i < n; i++) for (int j = 0; j < nu; j++) bG[i * nu + j] += brhs[i] * ut[j];
        for (int j = 0; j < nu; j++) { S acc = 0; for (int i = 0; i < n; i++) acc += sb[0].Gm[i * nu + j] * brhs[i]; du_t[j] += acc; }
        for (int i = 0; i < n; i++) bq[i] = 0;
        fields_vjp(P, g, sb[0], bG.data(), bdhdx_ext.data(), brhs.data(), bq.data());
        for (int i = 0; i < n; i++) bxin[i] += bq[i];
        // commit
        S* dut = du + (b * L + t) * nu; for (int j = 0; j < nu; j++) dut[j] = du_t[j];
        for (int i = 0; i < n; i++) bx[i] = bxin[i];
      }
      for (int i = 0; i < n; i++) gx0[b * n + i] = bx[i];
    }
  });
  (void)pf; (void)pf_stride;
}

// ================================================================== pybind entry points
#define GATHER_CONST \
  auto hw = cptrs<scalar_t>(hwv); auto hb = cptrs<scalar_t>(hbv); \
  auto jw = cptrs<scalar_t>(jwv); auto jb = cptrs<scalar_t>(jbv); \
  auto rw = cptrs<scalar_t>(rwv); auto rb = cptrs<scalar_t>(rbv); \
  auto gw = cptrs<scalar_t>(gwv); auto gb = cptrs<scalar_t>(gbv); \
  Params<scalar_t> P; \
  fill_params(P, hw, hb, jw, jb, rw, rb, glw.data_ptr<scalar_t>(), glb.data_ptr<scalar_t>(), gw, gb, \
              ow.defined() ? ow.data_ptr<scalar_t>() : nullptr, ob.defined() ? ob.data_ptr<scalar_t>() : nullptr); \
  P.n = n; P.nu = nu; P.ny = ny; P.nh = nh; P.K = K; P.out_linear = out_linear; P.has_bound = has_bound; \
  P.dt = (scalar_t)dt; P.bound = (scalar_t)bound; P.jr_scale = (scalar_t)jr_scale; P.g_scale = (scalar_t)g_scale;

void phnn_fwd(
    torch::Tensor x0, torch::Tensor u,
    std::vector<torch::Tensor> hwv, std::vector<torch::Tensor> hbv,
    std::vector<torch::Tensor> jwv, std::vector<torch::Tensor> jbv,
    std::vector<torch::Tensor> rwv, std::vector<torch::Tensor> rbv,
    torch::Tensor glw, torch::Tensor glb,
    std::vector<torch::Tensor> gwv, std::vector<torch::Tensor> gbv,
    torch::Tensor ow, torch::Tensor ob,
    torch::Tensor out, torch::Tensor xstates,
    int64_t n, int64_t nu, int64_t ny, int64_t nh, int64_t K,
    double dt, int64_t out_linear, int64_t has_bound, double bound, double jr_scale, double g_scale) {
  int64_t B = x0.size(0), L = u.size(1);
  AT_DISPATCH_FLOATING_TYPES(x0.scalar_type(), "phnn_fwd", [&] {
    GATHER_CONST
    fwd_impl(P, x0.data_ptr<scalar_t>(), u.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(), xstates.data_ptr<scalar_t>(), B, L);
  });
}

void phnn_bwd(
    torch::Tensor u, torch::Tensor xstates, torch::Tensor grad_out,
    std::vector<torch::Tensor> hwv, std::vector<torch::Tensor> hbv,
    std::vector<torch::Tensor> jwv, std::vector<torch::Tensor> jbv,
    std::vector<torch::Tensor> rwv, std::vector<torch::Tensor> rbv,
    torch::Tensor glw, torch::Tensor glb,
    std::vector<torch::Tensor> gwv, std::vector<torch::Tensor> gbv,
    torch::Tensor ow, torch::Tensor ob,
    // per-lane grads (leading B dim):
    std::vector<torch::Tensor> dhwv, std::vector<torch::Tensor> dhbv,
    std::vector<torch::Tensor> djwv, std::vector<torch::Tensor> djbv,
    std::vector<torch::Tensor> drwv, std::vector<torch::Tensor> drbv,
    torch::Tensor dglw, torch::Tensor dglb,
    std::vector<torch::Tensor> dgwv, std::vector<torch::Tensor> dgbv,
    torch::Tensor dow, torch::Tensor dob,
    torch::Tensor du, torch::Tensor gx0,
    int64_t n, int64_t nu, int64_t ny, int64_t nh, int64_t K,
    double dt, int64_t out_linear, int64_t has_bound, double bound, double jr_scale, double g_scale) {
  int64_t B = xstates.size(0), L = xstates.size(1);
  AT_DISPATCH_FLOATING_TYPES(xstates.scalar_type(), "phnn_bwd", [&] {
    GATHER_CONST
    // build per-lane Grads structs (pointers offset per lane)
    std::vector<std::vector<scalar_t*>> Ghw(B), Ghb(B), Gjw(B), Gjb(B), Grw(B), Grb(B), Ggw(B), Ggb(B);
    std::vector<Grads<scalar_t>> lane(B);
    auto lane_ptrs = [&](std::vector<torch::Tensor>& v, int64_t b) {
      std::vector<scalar_t*> p; for (auto& t : v) { int64_t per = t.numel() / B; p.push_back(t.data_ptr<scalar_t>() + b * per); } return p; };
    scalar_t* dglw_p = dglw.data_ptr<scalar_t>(); scalar_t* dglb_p = dglb.data_ptr<scalar_t>();
    int64_t glw_per = dglw.numel() / B, glb_per = dglb.numel() / B;
    scalar_t* dow_p = dow.defined() ? dow.data_ptr<scalar_t>() : nullptr;
    scalar_t* dob_p = dob.defined() ? dob.data_ptr<scalar_t>() : nullptr;
    int64_t ow_per = dow.defined() ? dow.numel() / B : 0, ob_per = dob.defined() ? dob.numel() / B : 0;
    for (int64_t b = 0; b < B; b++) {
      Ghw[b] = lane_ptrs(dhwv, b); Ghb[b] = lane_ptrs(dhbv, b);
      Gjw[b] = lane_ptrs(djwv, b); Gjb[b] = lane_ptrs(djbv, b);
      Grw[b] = lane_ptrs(drwv, b); Grb[b] = lane_ptrs(drbv, b);
      Ggw[b] = lane_ptrs(dgwv, b); Ggb[b] = lane_ptrs(dgbv, b);
      lane[b] = Grads<scalar_t>{Ghw[b].data(), Ghb[b].data(), Gjw[b].data(), Gjb[b].data(), Grw[b].data(), Grb[b].data(),
        dglw_p + b * glw_per, dglb_p + b * glb_per, Ggw[b].data(), Ggb[b].data(),
        dow_p ? dow_p + b * ow_per : nullptr, dob_p ? dob_p + b * ob_per : nullptr};
    }
    bwd_impl(P, u.data_ptr<scalar_t>(), xstates.data_ptr<scalar_t>(), grad_out.data_ptr<scalar_t>(),
             lane.data(), du.data_ptr<scalar_t>(), gx0.data_ptr<scalar_t>(), B, L, 0, (scalar_t*)nullptr);
  });
}
"""


def _get_extension():
    global _EXTENSION
    if _EXTENSION is None:
        from torch.utils.cpp_extension import load_inline

        cflags, ldflags = _build_flags()
        tag = hashlib.md5("".join((_SRC, *cflags, *ldflags)).encode()).hexdigest()[:10]
        _EXTENSION = load_inline(
            name=f"tsfast_phnn_c_{tag}",
            cpp_sources=_SRC,
            functions=["phnn_fwd", "phnn_bwd"],
            extra_cflags=cflags,
            extra_ldflags=ldflags,
        )
    return _EXTENSION


def check_rollout_args(spec: PHNNSpec, u: torch.Tensor, x0: torch.Tensor) -> None:
    if u.device.type != "cpu" or x0.device.type != "cpu":
        raise RuntimeError(f"the c backend requires cpu tensors, got {u.device.type}")
    if spec.n_state > 32 or spec.hidden > 512 or spec.num_layers > 4:
        raise RuntimeError(f"spec {spec} exceeds the c backend buffers (n_state<=32, hidden<=512, num_layers<=4)")
    if u.dim() != 3 or u.shape[-1] != spec.n_input:
        raise RuntimeError(f"expected u of shape [B, L, {spec.n_input}], got {tuple(u.shape)}")


def _scalars(core, spec: PHNNSpec, p: dict) -> dict:
    return dict(
        n=spec.n_state, nu=spec.n_input, ny=spec.n_output, nh=spec.hidden, K=spec.n_linear,
        dt=float(core.dt), out_linear=int(spec.output == "linear"),
        has_bound=int(spec.has_bound), bound=bound_value(core),
        jr_scale=float(core.jr_scale), g_scale=float(core.g_scale),
    )


def _cd(lst):
    return [t.detach().contiguous() for t in lst]


def _run_fwd(ext, spec, p, sc, u, x0):
    B, L = u.shape[0], u.shape[1]
    out = torch.empty(B, L, spec.n_output, dtype=u.dtype)
    xstates = torch.empty(B, L, spec.n_state, dtype=u.dtype)
    ow = p["ow"] if p["ow"] is not None else torch.empty(0, dtype=u.dtype)
    ob = p["ob"] if p["ob"] is not None else torch.empty(0, dtype=u.dtype)

    ext.phnn_fwd(
        x0.contiguous(), u.contiguous(),
        _cd(p["hw"]), _cd(p["hb"]), _cd(p["jw"]), _cd(p["jb"]), _cd(p["rw"]), _cd(p["rb"]),
        p["glw"].detach().contiguous(), p["glb"].detach().contiguous(), _cd(p["gw"]), _cd(p["gb"]),
        ow.detach().contiguous(), ob.detach().contiguous(), out, xstates,
        sc["n"], sc["nu"], sc["ny"], sc["nh"], sc["K"], sc["dt"], sc["out_linear"], sc["has_bound"],
        sc["bound"], sc["jr_scale"], sc["g_scale"],
    )
    return out, xstates


class _CPHNNRollout(torch.autograd.Function):
    @staticmethod
    def forward(ctx, core, spec, u, x0, *params):
        ext = _get_extension()
        p = params_of(core)
        sc = _scalars(core, spec, p)
        out, xstates = _run_fwd(ext, spec, p, sc, u, x0)
        ctx.core, ctx.spec, ctx.sc, ctx.ext = core, spec, sc, ext
        ctx.save_for_backward(u, xstates)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        core, spec, sc, ext = ctx.core, ctx.spec, ctx.sc, ctx.ext
        u, xstates = ctx.saved_tensors
        p = params_of(core)
        B, L = xstates.shape[0], xstates.shape[1]
        dt = u.dtype

        def lane_like(lst):
            return [torch.zeros(B, *t.shape, dtype=dt) for t in lst]

        dhw, dhb = lane_like(p["hw"]), lane_like(p["hb"])
        djw, djb = lane_like(p["jw"]), lane_like(p["jb"])
        drw, drb = lane_like(p["rw"]), lane_like(p["rb"])
        dgw, dgb = lane_like(p["gw"]), lane_like(p["gb"])
        dglw = torch.zeros(B, *p["glw"].shape, dtype=dt)
        dglb = torch.zeros(B, *p["glb"].shape, dtype=dt)
        has_out = p["ow"] is not None
        dow = torch.zeros(B, *p["ow"].shape, dtype=dt) if has_out else torch.empty(0, dtype=dt)
        dob = torch.zeros(B, *p["ob"].shape, dtype=dt) if has_out else torch.empty(0, dtype=dt)
        du = torch.zeros(B, L, spec.n_input, dtype=dt)
        gx0 = torch.zeros(B, spec.n_state, dtype=dt)
        ow = p["ow"] if has_out else torch.empty(0, dtype=dt)
        ob = p["ob"] if has_out else torch.empty(0, dtype=dt)

        ext.phnn_bwd(
            u.contiguous(), xstates.contiguous(), grad_out.contiguous(),
            _cd(p["hw"]), _cd(p["hb"]), _cd(p["jw"]), _cd(p["jb"]), _cd(p["rw"]), _cd(p["rb"]),
            p["glw"].detach().contiguous(), p["glb"].detach().contiguous(), _cd(p["gw"]), _cd(p["gb"]),
            ow.detach().contiguous(), ob.detach().contiguous(),
            dhw, dhb, djw, djb, drw, drb, dglw, dglb, dgw, dgb, dow, dob, du, gx0,
            sc["n"], sc["nu"], sc["ny"], sc["nh"], sc["K"], sc["dt"], sc["out_linear"], sc["has_bound"],
            sc["bound"], sc["jr_scale"], sc["g_scale"],
        )
        # sum per-lane grads to parameter grads, in flat_params order
        grads = []
        for lst in (dhw, dhb, djw, djb, drw, drb):
            grads += [g.sum(0) for g in lst]
        grads += [dglw.sum(0), dglb.sum(0)]
        for lst in (dgw, dgb):
            grads += [g.sum(0) for g in lst]
        if has_out:
            grads += [dow.sum(0), dob.sum(0)]
        du_out = du if ctx.needs_input_grad[2] else None
        dx0_out = gx0 if ctx.needs_input_grad[3] else None
        return (None, None, du_out, dx0_out, *grads)


def c_rollout(core, spec: PHNNSpec, u: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
    """Run the PHNN section rollout through the generic C++ extension (autograd-capable).

    ``u`` is the future input ``[B, L, n_input]`` and ``x0`` the encoder state
    ``[B, n_state]``; returns the output sequence ``[B, L, n_output]`` for the L future
    steps (the encoder warm-up is prepended by the caller).
    """
    check_rollout_args(spec, u, x0)
    from .common import flat_params

    params = flat_params(core)
    if not torch.is_grad_enabled() or not any(t.requires_grad for t in [u, x0, *params]):
        ext = _get_extension()
        p = params_of(core)
        out, _ = _run_fwd(ext, spec, p, _scalars(core, spec, p), u, x0)
        return out
    return _CPHNNRollout.apply(core, spec, u, x0, *params)
