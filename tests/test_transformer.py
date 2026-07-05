"""Tests for tsfast.models.architectures.transformer (TSTransformer)."""

import sys
from pathlib import Path

import pytest
import torch

from tsfast.models.architectures.transformer import TSTransformer


def _model(**kwargs):
    defaults = dict(n_input=2, n_output=3, n_init=8, d_model=16, n_heads=2, n_layers=2, n_in=4, chunk_len=12)
    defaults.update(kwargs)
    return TSTransformer(**defaults)


class TestTSTransformer:
    def test_output_shape_and_warmup_zeros(self):
        torch.manual_seed(0)
        m = _model().eval()
        out = m(torch.randn(5, 30, 5))
        assert out.shape == (5, 30, 3)
        assert out[:, :8].abs().max() == 0

    def test_train_mode_returns_mean_and_logvar(self):
        torch.manual_seed(1)
        m = _model().train()
        mean, logvar = m(torch.randn(2, 20, 5))
        assert mean.shape == logvar.shape == (2, 20, 3)

    def test_only_warmup_y_is_read(self):
        """Measured outputs beyond n_init must not influence predictions (no leakage)."""
        torch.manual_seed(2)
        m = _model().eval()
        xin = torch.randn(2, 30, 5)
        perturbed = xin.clone()
        perturbed[:, 8:, 2:] = torch.randn(2, 22, 3)
        with torch.no_grad():
            assert torch.equal(m(xin), m(perturbed))

    def test_chunked_prefix_matches_single_pass(self):
        """The first chunk of a long simulation equals the single-pass result on the prefix."""
        torch.manual_seed(3)
        m = _model().eval()
        xin = torch.randn(2, 8 + 12 * 3, 5)
        with torch.no_grad():
            full = m(xin)
            prefix = m(xin[:, : 8 + 12])
        assert torch.allclose(full[:, : 8 + 12], prefix, atol=1e-6)

    def test_chunk_feedback_carries_predictions(self):
        """Later chunks must depend on earlier predictions through the initial-condition tokens."""
        torch.manual_seed(4)
        m = _model().eval()
        xin = torch.randn(1, 8 + 12 * 2, 5)
        perturbed = xin.clone()
        perturbed[:, 8 : 8 + 12, :2] += 1.0  # change first-chunk inputs only
        with torch.no_grad():
            a, b = m(xin), m(perturbed)
        assert not torch.allclose(a[:, 8 + 12 :], b[:, 8 + 12 :])

    def test_recurrent_patching_engages(self):
        torch.manual_seed(5)
        m = _model(n_init=30, n_in=4, max_ctx_tokens=8).eval()
        out = m(torch.randn(2, 50, 5))
        assert out.shape == (2, 50, 3)

    @pytest.mark.parametrize(
        "max_ctx_tokens,unused",
        [
            (100, ("rnn_patch.", "encoder_wte_patch.")),  # short context: linear embedding
            (8, ("encoder_wte.",)),  # long context: recurrent patching
        ],
    )
    def test_gradients_flow_everywhere(self, max_ctx_tokens, unused):
        torch.manual_seed(6)
        m = _model(n_init=30, n_in=4, max_ctx_tokens=max_ctx_tokens).train()
        mean, logvar = m(torch.randn(3, 45, 5))
        (mean[:, 30:].pow(2).mean() + logvar[:, 30:].pow(2).mean()).backward()
        missing = [
            n
            for n, p in m.named_parameters()
            if not n.startswith(unused) and (p.grad is None or p.grad.abs().sum() == 0)
        ]
        assert not missing, f"no gradient for: {missing}"

    def test_parameter_validation(self):
        with pytest.raises(ValueError, match="n_in"):
            _model(n_in=8)  # n_in == n_init
        with pytest.raises(ValueError, match="chunk_len"):
            _model(chunk_len=2)  # smaller than n_in

    def test_short_sequence_rejected(self):
        with pytest.raises(ValueError, match="too short"):
            _model()(torch.randn(2, 8, 5))


class TestReferenceEquivalence:
    """Regression version of ``comparisons/compare_transformer.py``.

    The reference repository is unlicensed, so the module is downloaded from a
    pinned commit at test time instead of being transcribed; the tests skip
    when the download fails (offline CI).
    """

    @pytest.fixture(scope="class")
    def compare(self):
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "comparisons"))
        import compare_transformer as ct

        try:
            ref_mod = ct.load_reference()
        except OSError as e:
            pytest.skip(f"reference download unavailable: {e}")
        orig_mask = torch.nn.Transformer.generate_square_subsequent_mask
        ct.force_float64_masks()
        yield lambda label, **kwargs: ct.compare(ref_mod, label, **kwargs)
        torch.nn.Transformer.generate_square_subsequent_mask = orig_mask

    def test_linear_context_path(self, compare):
        assert compare("linear context", n_u=2, n_y=3, m=30, n_in=5, n_query=40, d_model=16, n_heads=2, n_layers=2)

    def test_recurrent_patching_path(self, compare):
        assert compare("recurrent patching", n_u=1, n_y=1, m=810, n_in=5, n_query=24, d_model=16, n_heads=2, n_layers=2)
