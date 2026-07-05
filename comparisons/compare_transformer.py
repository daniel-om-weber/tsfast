"""Compare tsfast's TSTransformer against the reference implementation of Rufolo et al.

Reference: the ``TSTransformer`` class from ``transformer_sim.py`` of
github.com/mattrufolo/sysid-prob-transformer, the implementation published with
"Enhanced Transformer architecture for in-context learning of dynamical systems"
(Rufolo, Piga, Maroni & Forgione, ECC 2025, arXiv:2410.03291), which extends the
MIT-licensed github.com/forgi86/sysid-transformers. The repository declares no
license and is not packaged for PyPI, so nothing is vendored here: the module is
downloaded at run time from the pinned commit below and imported from a
temporary directory (its unused local ``metrics`` import is stubbed).

Parameter mapping (tsfast -> reference): the encoder/decoder block trees are
name-identical (``blocks.N.ln_1/self_attn.mha/ln_2/cross_attn.mha/ln_3/mlp``,
final ``ln_f``); the embeddings rename as ``encoder_wte -> encoder_wte_noRNN``
(linear context path), ``encoder_wte_patch -> encoder_wte`` and
``rnn_patch -> RNN`` (recurrent-patching path), ``decoder_wte_init ->
decoder_wte1``, ``decoder_wte_new -> decoder_wte2``, ``head_mean/head_logvar ->
lm_head_mean/lm_head_logvar``. Sinusoidal positional-encoding buffers are
deterministic and excluded from the transplant (they differ only in ``max_len``);
the reference's ``encoder_wte2`` is unused by its forward and stays untouched.

The data interface differs by design: the reference takes separate context,
query, and initial-condition tensors, while tsfast reads one
``prediction_concat`` ``[u, y]`` tensor with ``n_init = m + n_in`` warm-up
samples. Both orderings of the internal concatenations match, so outputs must
agree exactly. The reference hardcodes its patching threshold at 400 context
samples; tsfast's ``max_ctx_tokens`` is set to 400 accordingly.

For every configuration the script reports the maximum relative deviation of
the predicted mean, predicted standard deviation, and all shared parameter
gradients, in float64, for both context-embedding paths. Expected agreement:
< 1e-12.
"""

import sys
import tempfile
import types
import urllib.request
from pathlib import Path

import torch

from tsfast.models.architectures.transformer import TSTransformer

TOL = 1e-12
REF_COMMIT = "8ee7d17f09156b0103efb7f70d104b29914011fd"
REF_URL = f"https://raw.githubusercontent.com/mattrufolo/sysid-prob-transformer/{REF_COMMIT}/transformer_sim.py"

# tsfast parameter-name prefix -> reference prefix; block trees map one-to-one.
NAME_MAP = {
    "rnn_patch.": "RNN.",
    "encoder_wte_patch.": "encoder_wte.",
    "encoder_wte.": "encoder_wte_noRNN.",
    "decoder_wte_init.": "decoder_wte1.",
    "decoder_wte_new.": "decoder_wte2.",
    "head_mean.": "lm_head_mean.",
    "head_logvar.": "lm_head_logvar.",
}


def rel(a, b):
    return (a - b).abs().max().item() / (b.abs().max().item() + 1e-30)


def load_reference():
    """Download the pinned reference module and import it with a stubbed ``metrics``."""
    tmp = Path(tempfile.mkdtemp(prefix="sysid_prob_transformer_"))
    path = tmp / "transformer_sim.py"
    path.write_bytes(urllib.request.urlopen(REF_URL, timeout=30).read())
    sys.modules.setdefault("metrics", types.ModuleType("metrics"))
    module = types.ModuleType("transformer_sim_ref")
    exec(compile(path.read_text(), str(path), "exec"), module.__dict__)
    return module


def force_float64_masks():
    """Make the reference's causal masks float64; torch rejects its float32 masks inside a float64 model.

    The mask holds only 0/-inf, so this changes no values.
    """
    orig = torch.nn.Transformer.generate_square_subsequent_mask
    torch.nn.Transformer.generate_square_subsequent_mask = staticmethod(
        lambda sz, device=None, dtype=None: orig(sz, device=device, dtype=torch.float64)
    )


def map_name(name: str) -> str:
    for ours, theirs in NAME_MAP.items():
        if name.startswith(ours):
            return theirs + name[len(ours) :]
    return name


def transplant(model: TSTransformer, ref: torch.nn.Module):
    """Copy tsfast parameters into the reference model; PE buffers stay deterministic."""
    state = {map_name(k): v for k, v in model.state_dict().items() if not k.endswith("wpe.pe")}
    missing, unexpected = ref.load_state_dict(state, strict=False)
    assert not unexpected, unexpected
    leftovers = [m for m in missing if not (m.endswith("wpe.pe") or m.startswith("encoder_wte2."))]
    assert not leftovers, leftovers


def compare(ref_mod, label, n_u, n_y, m, n_in, n_query, d_model, n_heads, n_layers, d_rnn=16):
    torch.manual_seed(0)
    model = TSTransformer(
        n_u,
        n_y,
        n_init=m + n_in,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        n_in=n_in,
        chunk_len=n_query - n_in,
        max_ctx_tokens=400,  # the reference hardcodes this threshold
        d_rnn=d_rnn,
    ).double()
    cfg = ref_mod.Config(
        n_layer=n_layers, n_head=n_heads, n_embd=d_model, n_u=n_u, n_y=n_y, d_model_RNN=d_rnn, dropout=0.0, bias=False
    )
    ref = ref_mod.TSTransformer(cfg).double()
    transplant(model, ref)

    B = 3
    u_ctx, y_ctx = torch.randn(B, m, n_u).double(), torch.randn(B, m, n_y).double()
    u_new, y_new = torch.randn(B, n_query, n_u).double(), torch.randn(B, n_query, n_y).double()

    y_masked = torch.cat((y_ctx, y_new[:, :n_in], torch.zeros(B, n_query - n_in, n_y).double()), dim=1)
    x = torch.cat((torch.cat((u_ctx, u_new), dim=1), y_masked), dim=-1)

    model.train()
    mean, logvar = model(x)
    mean, std = mean[:, m + n_in :], (logvar[:, m + n_in :] / 2).exp()
    mean_ref, std_ref, _, _ = ref(y_ctx, u_ctx, u_new, y_new, n_in)

    (mean.pow(2).mean() + std.pow(2).mean()).backward()
    (mean_ref.pow(2).mean() + std_ref.pow(2).mean()).backward()

    devs = {"mean": rel(mean, mean_ref), "std": rel(std, std_ref)}
    ref_grads = {n: p.grad for n, p in ref.named_parameters()}
    grad_devs = []
    for n, p in model.named_parameters():
        g_ref = ref_grads[map_name(n)]
        assert (p.grad is None) == (g_ref is None), n
        if p.grad is not None and p.grad.abs().sum() > 0:
            grad_devs.append(rel(p.grad, g_ref))
    devs["grads"] = max(grad_devs)

    ok = all(v < TOL for v in devs.values())
    print(f"{label:<42} mean {devs['mean']:.2e}  std {devs['std']:.2e}  grads {devs['grads']:.2e}  {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    ref_mod = load_reference()
    force_float64_masks()
    results = [
        compare(ref_mod, "SISO, linear context", n_u=1, n_y=1, m=60, n_in=10, n_query=110, d_model=32, n_heads=4, n_layers=3),
        compare(ref_mod, "MIMO, linear context", n_u=2, n_y=3, m=100, n_in=10, n_query=64, d_model=64, n_heads=4, n_layers=2),
        compare(ref_mod, "SISO, recurrent patching (m=900)", n_u=1, n_y=1, m=900, n_in=10, n_query=64, d_model=32, n_heads=2, n_layers=2),
        compare(ref_mod, "paper config (12 layers, d128, h4)", n_u=1, n_y=1, m=400, n_in=10, n_query=100, d_model=128, n_heads=4, n_layers=12, d_rnn=128),
    ]
    sys.exit(0 if all(results) else 1)
