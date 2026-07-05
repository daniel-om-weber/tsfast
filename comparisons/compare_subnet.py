"""Compare tsfast's SUBNET model against the deepSI subspace-encoder reference.

Reference: Beintema, Schoukens & Tóth, "Deep Subspace Encoders for Nonlinear System
Identification" (Automatica 156:111210, 2023, arXiv:2210.14816); reference
implementation ``SS_encoder_general`` from github.com/GerbenBeintema/deepSI, branch
``legacy`` (BSD-3-Clause). The repository is not packaged for current PyPI, so the
needed pieces are transcribed verbatim below: ``feed_forward_nn``, ``simple_res_net``,
``default_encoder_net`` (``deepSI/utils/torch_nets.py``, ``fit_systems/encoders.py``)
and the section rollout semantics of ``SS_encoder_general.loss`` (output observed from
the current state *before* the state update).

Architectural note: tsfast's ``SubnetSSM`` follows deepSI's encoder and training scheme
exactly, but deviates from the SUBNET *defaults* by design in the dynamics components,
so the fused ``NeuralStateSpace`` rollout backends apply: the state transition is a
plain MLP on ``[x, u]`` (deepSI default: linear + MLP residual net) and the output map
is linear (deepSI default: residual net). The comparison therefore instantiates the
deepSI framework with matching component nets — a plain ``feed_forward_nn`` transition
and a linear output — which the framework supports through its ``f_net``/``h_net``
arguments. The encoder is deepSI's unmodified default. Parameters are copied 1:1
(``net_lin``/``net_non_lin`` of ``simple_res_net`` map to ``lin``/``mlp`` of tsfast's
``ResMLP``).

For every configuration the script reports the maximum relative deviation of the
full-section output (encoder warm-up + rollout) and of all parameter and input
gradients, in float64, including MIMO and na != nb encoder windows. Expected
agreement: < 1e-12.
"""

import sys

import numpy as np
import torch
from torch import nn

from tsfast.models.architectures.subnet import SubnetSSM

TOL = 1e-12


def rel(a, b):
    return (a - b).abs().max().item() / (b.abs().max().item() + 1e-30)


# --- transcribed from deepSI (BSD-3-Clause), deepSI/utils/torch_nets.py ---
class feed_forward_nn(nn.Module):
    def __init__(self, n_in=6, n_out=5, n_nodes_per_layer=64, n_hidden_layers=2, activation=nn.Tanh):
        super(feed_forward_nn, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        seq = [nn.Linear(n_in, n_nodes_per_layer), activation()]
        assert n_hidden_layers > 0
        for i in range(n_hidden_layers - 1):
            seq.append(nn.Linear(n_nodes_per_layer, n_nodes_per_layer))
            seq.append(activation())
        seq.append(nn.Linear(n_nodes_per_layer, n_out))
        self.net = nn.Sequential(*seq)
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, val=0)

    def forward(self, X):
        return self.net(X)


class simple_res_net(nn.Module):
    def __init__(self, n_in=6, n_out=5, n_nodes_per_layer=64, n_hidden_layers=2, activation=nn.Tanh):
        super(simple_res_net, self).__init__()
        self.net_lin = nn.Linear(n_in, n_out)
        self.n_in = n_in
        self.n_out = n_out
        if n_hidden_layers > 0:
            self.net_non_lin = feed_forward_nn(
                n_in, n_out, n_nodes_per_layer=n_nodes_per_layer, n_hidden_layers=n_hidden_layers, activation=activation
            )
        else:
            self.net_non_lin = None

    def forward(self, x):
        if self.net_non_lin is not None:
            return self.net_lin(x) + self.net_non_lin(x)
        return self.net_lin(x)


# --- transcribed from deepSI (BSD-3-Clause), deepSI/fit_systems/encoders.py ---
class default_encoder_net(nn.Module):
    def __init__(self, nb, nu, na, ny, nx, n_nodes_per_layer=64, n_hidden_layers=2, activation=nn.Tanh):
        super(default_encoder_net, self).__init__()
        self.nu = tuple() if nu is None else ((nu,) if isinstance(nu, int) else nu)
        self.ny = tuple() if ny is None else ((ny,) if isinstance(ny, int) else ny)
        self.net = simple_res_net(
            n_in=nb * np.prod(self.nu, dtype=int) + na * np.prod(self.ny, dtype=int),
            n_out=nx,
            n_nodes_per_layer=n_nodes_per_layer,
            n_hidden_layers=n_hidden_layers,
            activation=activation,
        )

    def forward(self, upast, ypast):
        net_in = torch.cat([upast.view(upast.shape[0], -1), ypast.view(ypast.shape[0], -1)], axis=1)
        return self.net(net_in)


class MLPStateNet(nn.Module):
    """deepSI-style f_net with a plain MLP on [x, u], matching tsfast's transition."""

    def __init__(self, nx, nu, n_nodes_per_layer, n_hidden_layers):
        super().__init__()
        self.net = feed_forward_nn(nx + nu, nx, n_nodes_per_layer=n_nodes_per_layer, n_hidden_layers=n_hidden_layers)

    def forward(self, x, u):
        return self.net(torch.cat([x, u.view(u.shape[0], -1)], axis=1))


class ReferenceSubnet(nn.Module):
    """deepSI SS_encoder_general rollout with encoder + f_net + linear h_net.

    Follows ``SS_encoder_general.loss``: for each future step the output observes the
    current state (``hn(x)``) before ``x = fn(x, u)`` advances it, starting from the
    encoder state — the same ordering tsfast's ``SubnetSSM`` implements.
    """

    def __init__(self, nu, ny, nx, na, nb, n_init, hidden_size, num_layers):
        super().__init__()
        self.nu, self.ny, self.n_init, self.na, self.nb = nu, ny, n_init, na, nb
        self.encoder = default_encoder_net(nb=nb, nu=nu, na=na, ny=ny, nx=nx)
        self.fn = MLPStateNet(nx, nu, hidden_size, num_layers)
        self.hn = nn.Linear(nx, ny)

    def forward(self, xin):
        # contiguous: the verbatim deepSI encoder flattens with .view, which channel slices don't support
        u, ymeas = xin[..., : self.nu].contiguous(), xin[..., self.nu :].contiguous()
        n0 = self.n_init
        x = self.encoder(u[:, n0 - self.nb : n0].contiguous(), ymeas[:, n0 - self.na : n0].contiguous())
        outs = []
        for t in range(n0, xin.shape[1]):
            outs.append(self.hn(x))
            x = self.fn(x, u[:, t])
        warmup = xin.new_zeros(xin.shape[0], n0, self.ny)
        return torch.cat((warmup, torch.stack(outs, dim=1)), dim=1)


def _n_linear(seq):
    return len([m for m in seq if isinstance(m, nn.Linear)])


def transplant(model: SubnetSSM, ref: ReferenceSubnet):
    sd = {}
    for i in range(_n_linear(ref.fn.net.net)):
        for wb in ("weight", "bias"):
            sd[f"core.net.{2 * i}.{wb}"] = getattr(ref.fn.net.net[2 * i], wb)
    for i in range(_n_linear(ref.encoder.net.net_non_lin.net)):
        for wb in ("weight", "bias"):
            sd[f"encoder.net.mlp.{2 * i}.{wb}"] = getattr(ref.encoder.net.net_non_lin.net[2 * i], wb)
    for wb in ("weight", "bias"):
        sd[f"core.output_map.{wb}"] = getattr(ref.hn, wb)
        sd[f"encoder.net.lin.{wb}"] = getattr(ref.encoder.net.net_lin, wb)
    missing, unexpected = model.load_state_dict({k: v.detach().clone() for k, v in sd.items()}, strict=True)
    assert not missing and not unexpected, (missing, unexpected)


def run(model, xin):
    for p in model.parameters():
        p.grad = None
    xin = xin.clone().requires_grad_()
    out = model(xin)
    loss = (out**2).mean() + out.abs().sum() * 0.01
    loss.backward()
    return out.detach(), [p.grad.clone() for p in model.parameters()], xin.grad.clone()


CONFIGS = [
    dict(n_input=1, n_output=1, n_state=4, hidden_size=16, num_layers=2),
    dict(n_input=1, n_output=1, n_state=8, hidden_size=32, num_layers=1),
    dict(n_input=2, n_output=3, n_state=6, hidden_size=16, num_layers=2),
    dict(n_input=1, n_output=1, n_state=4, hidden_size=16, num_layers=2, na=7, nb=3),
]

torch.manual_seed(0)
failed = False
for cfg in CONFIGS:
    n_init = 8
    model = SubnetSSM(n_init=n_init, backend="eager", **cfg).double()
    ref = ReferenceSubnet(
        nu=cfg["n_input"], ny=cfg["n_output"], nx=cfg["n_state"], na=cfg.get("na", n_init), nb=cfg.get("nb", n_init),
        n_init=n_init, hidden_size=cfg["hidden_size"], num_layers=cfg["num_layers"],
    ).double()
    for p in ref.parameters():
        nn.init.normal_(p, std=0.4)
    transplant(model, ref)

    B, L = 5, n_init + 40
    xin = torch.randn(B, L, cfg["n_input"] + cfg["n_output"], dtype=torch.float64)
    out_t, grads_t, du_t = run(model, xin)
    out_r, grads_r, du_r = run(ref, xin)

    # align parameter order via the same transplant mapping: compare by name pairs
    names_t = [n for n, _ in model.named_parameters()]
    grads_t_by_name = dict(zip(names_t, grads_t))
    names_r = [n for n, _ in ref.named_parameters()]
    grads_r_by_name = dict(zip(names_r, grads_r))
    pairs = []
    for i in range(_n_linear(ref.fn.net.net)):
        for wb in ("weight", "bias"):
            pairs.append((f"core.net.{2 * i}.{wb}", f"fn.net.net.{2 * i}.{wb}"))
    for i in range(_n_linear(ref.encoder.net.net_non_lin.net)):
        for wb in ("weight", "bias"):
            pairs.append((f"encoder.net.mlp.{2 * i}.{wb}", f"encoder.net.net_non_lin.net.{2 * i}.{wb}"))
    for wb in ("weight", "bias"):
        pairs.append((f"core.output_map.{wb}", f"hn.{wb}"))
        pairs.append((f"encoder.net.lin.{wb}", f"encoder.net.net_lin.{wb}"))

    devs = [rel(out_t, out_r), rel(du_t, du_r)]
    devs += [rel(grads_t_by_name[a], grads_r_by_name[b]) for a, b in pairs]
    worst = max(devs)
    status = "OK  " if worst < TOL else "FAIL"
    failed |= worst >= TOL
    print(f"{status} {cfg}: output {devs[0]:.2e}, worst grad {max(devs[1:]):.2e}")

print("\nall configurations within tolerance" if not failed else "\nTOLERANCE EXCEEDED")
sys.exit(1 if failed else 0)
