"""Encoder-decoder Transformer for multi-step simulation of dynamical systems.

Reference: Rufolo, Piga & Forgione, "Enhanced Transformer architecture for
in-context learning of dynamical systems", ECC 2025 (arXiv:2410.03291);
reference implementation github.com/mattrufolo/sysid-prob-transformer
(itself based on nanoGPT). The blocks follow the reference: pre-LN
GPT-2-style layers without biases, sinusoidal positional encodings, a linear
context embedding replaced by recurrent patching for long contexts, a
dedicated linear embedding for the initial-condition samples prepended to
the decoder query, and a probabilistic output head predicting mean and
log-variance of a diagonal Gaussian.

Deviation by design: the reference is a meta-model trained across a system
class, with the context drawn from a separate record of the query system.
Here the model is trained on a single system, so the context is the
benchmark's initial-condition window, and signals longer than one query
window are simulated chunk by chunk, each chunk seeded with the last
``n_in`` predictions of the previous one (the decoder equivalent of
carrying a recurrent state).
"""

__all__ = [
    "PositionalEncoding",
    "TransformerEncoder",
    "TransformerDecoder",
    "TSTransformer",
]

import math

import torch
from torch import nn
from torch.nn import functional as F


class _LayerNorm(nn.Module):
    """LayerNorm with optional bias (PyTorch has no ``bias=False``)."""

    def __init__(self, ndim: int, bias: bool):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class _SelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0, causal: bool = True, bias: bool = False):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, n_heads, bias=bias, dropout=dropout, batch_first=True)
        self.causal = causal
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x):
        if self.causal:
            mask = nn.Transformer.generate_square_subsequent_mask(x.shape[1], device=x.device, dtype=x.dtype)
            x = self.mha(x, x, x, attn_mask=mask, is_causal=True, need_weights=False)[0]
        else:
            x = self.mha(x, x, x, need_weights=False)[0]
        return self.resid_dropout(x)


class _CrossAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0, bias: bool = False):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, n_heads, bias=bias, dropout=dropout, batch_first=True)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x, mem):
        x = self.mha(x, mem, mem, need_weights=False)[0]
        return self.resid_dropout(x)


class _MLP(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.0, bias: bool = False):
        super().__init__()
        self.c_fc = nn.Linear(d_model, 4 * d_model, bias=bias)
        self.c_proj = nn.Linear(4 * d_model, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.c_proj(F.gelu(self.c_fc(x))))


class PositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding added to the embeddings."""

    def __init__(self, d_model: int, dropout: float = 0.0, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[: x.size(1), :])


class _EncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0, bias: bool = False):
        super().__init__()
        self.ln_1 = _LayerNorm(d_model, bias)
        self.self_attn = _SelfAttention(d_model, n_heads, dropout, causal=False, bias=bias)
        self.ln_2 = _LayerNorm(d_model, bias)
        self.mlp = _MLP(d_model, dropout, bias)

    def forward(self, x):
        x = x + self.self_attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class _DecoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0, bias: bool = False):
        super().__init__()
        self.ln_1 = _LayerNorm(d_model, bias)
        self.self_attn = _SelfAttention(d_model, n_heads, dropout, causal=True, bias=bias)
        self.ln_2 = _LayerNorm(d_model, bias)
        self.cross_attn = _CrossAttention(d_model, n_heads, dropout, bias=bias)
        self.ln_3 = _LayerNorm(d_model, bias)
        self.mlp = _MLP(d_model, dropout, bias)

    def forward(self, x, mem):
        x = x + self.self_attn(self.ln_1(x))
        x = x + self.cross_attn(self.ln_2(x), mem)
        x = x + self.mlp(self.ln_3(x))
        return x


class TransformerEncoder(nn.Module):
    """Stack of non-causal pre-LN attention blocks with a final LayerNorm."""

    def __init__(self, d_model: int, n_heads: int, n_layers: int, dropout: float = 0.0, bias: bool = False):
        super().__init__()
        self.blocks = nn.ModuleList([_EncoderLayer(d_model, n_heads, dropout, bias) for _ in range(n_layers)])
        self.ln_f = _LayerNorm(d_model, bias)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return self.ln_f(x)


class TransformerDecoder(nn.Module):
    """Stack of causal pre-LN blocks with cross-attention and a final LayerNorm."""

    def __init__(self, d_model: int, n_heads: int, n_layers: int, dropout: float = 0.0, bias: bool = False):
        super().__init__()
        self.blocks = nn.ModuleList([_DecoderLayer(d_model, n_heads, dropout, bias) for _ in range(n_layers)])
        self.ln_f = _LayerNorm(d_model, bias)

    def forward(self, x, mem):
        for block in self.blocks:
            x = block(x, mem)
        return self.ln_f(x)


class TSTransformer(nn.Module):
    """Encoder-decoder simulation Transformer over ``[u, y]`` input channels.

    The input tensor carries ``n_input`` input channels followed by
    ``n_output`` measured-output channels (the ``prediction_concat`` layout).
    Only the first ``n_init`` steps of the output channels are ever read: the
    encoder embeds samples ``[0, n_init - n_in)`` as the context, the last
    ``n_in`` warm-up samples become the decoder's initial-condition tokens,
    and predictions start at ``n_init``. Earlier positions are zero and must
    be excluded from the loss via ``n_skip=n_init``.

    In training mode the forward returns ``(mean, logvar)`` full-length
    tensors for the Gaussian negative log-likelihood of the reference; in
    eval mode it returns the mean only, so metrics and inference wrappers see
    a plain simulation model. Sequences whose prediction span exceeds
    ``chunk_len`` are decoded chunk by chunk, each chunk re-seeded with the
    last ``n_in`` inputs and predictions, so arbitrarily long test signals
    stay within the positional-encoding range seen during training —
    ``chunk_len`` should therefore match the training prediction span.

    Contexts longer than ``max_ctx_tokens`` are compressed by the reference's
    recurrent patching: fixed ``max_ctx_tokens`` patches, each summarized by
    the final hidden state of a shared vanilla RNN.

    Args:
        n_input: exogenous input dimension (``u`` channels of the input tensor).
        n_output: observed output dimension (``y`` channels of the input tensor).
        n_init: warm-up length consumed by context and initial conditions;
            predictions and loss start here.
        d_model: embedding width of the attention blocks.
        n_heads: attention heads per block.
        n_layers: number of encoder and of decoder blocks.
        n_in: initial-condition samples prepended to the decoder query.
        chunk_len: maximum prediction span per decoder pass.
        max_ctx_tokens: context length above which recurrent patching engages,
            and the fixed number of patches it produces.
        d_rnn: hidden width of the patching RNN.
        dropout: dropout probability in all blocks.
        bias: use biases in linear layers and LayerNorms (GPT-2 style).
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_init: int = 50,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        n_in: int = 10,
        chunk_len: int = 128,
        max_ctx_tokens: int = 400,
        d_rnn: int = 128,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()
        if not 0 < n_in < n_init:
            raise ValueError(f"n_in={n_in} must be positive and smaller than n_init={n_init}")
        if chunk_len < n_in:
            raise ValueError(f"chunk_len={chunk_len} must be at least n_in={n_in}")
        self.n_input, self.n_output, self.n_init = n_input, n_output, n_init
        self.n_in, self.chunk_len, self.max_ctx_tokens = n_in, chunk_len, max_ctx_tokens

        self.encoder = TransformerEncoder(d_model, n_heads, n_layers, dropout, bias)
        self.decoder = TransformerDecoder(d_model, n_heads, n_layers, dropout, bias)

        self.rnn_patch = nn.RNN(n_input + n_output, d_rnn, num_layers=1, batch_first=True)
        self.encoder_wte_patch = nn.Linear(d_rnn, d_model)
        self.encoder_wte = nn.Linear(n_input + n_output, d_model)
        self.encoder_wpe = PositionalEncoding(d_model, dropout, max_len=max(max_ctx_tokens, n_init))
        self.decoder_wte_init = nn.Linear(n_input + n_output, d_model)
        self.decoder_wte_new = nn.Linear(n_input, d_model)
        self.decoder_wpe = PositionalEncoding(d_model, dropout, max_len=n_in + chunk_len)

        self.head_mean = nn.Linear(d_model, n_output, bias=True)
        self.head_logvar = nn.Linear(d_model, n_output, bias=True)

    def embed_ctx(self, u_ctx: torch.Tensor, y_ctx: torch.Tensor) -> torch.Tensor:
        """Embed the context ``(u, y)`` samples, recurrently patched when long."""
        yu = torch.cat((y_ctx, u_ctx), dim=-1)
        if yu.shape[1] > self.max_ctx_tokens:
            # Fixed number of patches: drop the leading remainder, summarize each
            # patch by the RNN's final hidden state (reference patching scheme).
            B = yu.shape[0]
            patch_len = yu.shape[1] // self.max_ctx_tokens
            yu = yu[:, yu.shape[1] - patch_len * self.max_ctx_tokens :]
            _, hn = self.rnn_patch(yu.reshape(B * self.max_ctx_tokens, patch_len, -1))
            tok = self.encoder_wte_patch(hn[-1].view(B, self.max_ctx_tokens, -1))
        else:
            tok = self.encoder_wte(yu)
        return self.encoder_wpe(tok)

    def decode_chunk(
        self, mem: torch.Tensor, u_init: torch.Tensor, y_init: torch.Tensor, u_new: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One decoder pass: ``n_in`` initial-condition tokens followed by the query inputs."""
        tok_init = self.decoder_wte_init(torch.cat((u_init, y_init), dim=-1))
        tok_new = self.decoder_wte_new(u_new)
        tgt = self.decoder_wpe(torch.cat((tok_init, tok_new), dim=1))
        h = self.decoder(tgt, mem)[:, u_init.shape[1] :]
        return self.head_mean(h), self.head_logvar(h)

    def forward(self, x: torch.Tensor):
        """Simulate from the warm-up window; returns ``[B, L, n_output]`` with zeros before ``n_init``.

        Training mode returns ``(mean, logvar)``, eval mode the mean only.
        """
        if x.shape[1] <= self.n_init:
            raise ValueError(f"sequence length {x.shape[1]} too short for warm-up n_init={self.n_init}")
        u, y = x[..., : self.n_input], x[..., self.n_input :]
        n0, n_in = self.n_init, self.n_in

        mem = self.encoder(self.embed_ctx(u[:, : n0 - n_in], y[:, : n0 - n_in]))
        u_init, y_init = u[:, n0 - n_in : n0], y[:, n0 - n_in : n0]

        means, logvars = [], []
        for pos in range(n0, x.shape[1], self.chunk_len):
            end = min(pos + self.chunk_len, x.shape[1])
            mean, logvar = self.decode_chunk(mem, u_init, y_init, u[:, pos:end])
            means.append(mean)
            logvars.append(logvar)
            u_init, y_init = u[:, end - n_in : end], mean[:, -n_in:]

        warmup = x.new_zeros(x.shape[0], n0, self.n_output)
        mean_full = torch.cat([warmup, *means], dim=1)
        if self.training:
            return mean_full, torch.cat([warmup, *logvars], dim=1)
        return mean_full
