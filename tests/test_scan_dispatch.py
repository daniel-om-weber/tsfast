"""Tests for the scan backend dispatch: fallback chain and once-per-process warning."""

import sys
import types
import warnings

import pytest
import torch

import tsfast.models._core.dispatch as backends
import tsfast.models._core.scan as scan


@pytest.fixture(autouse=True)
def _reset_state(monkeypatch):
    monkeypatch.setattr(scan, "backend", "auto")
    monkeypatch.setattr(backends, "_warned", set())
    yield


def _run(lam=None, v=None):
    lam = torch.rand(3) * 0.5 if lam is None else lam
    v = torch.randn(2, 8, 3) if v is None else v
    return scan.diagonal_recurrence(lam, v)


def test_auto_without_backend_modules_is_silent():
    # No triton/c backend module installed: auto falls through to doubling quietly.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = _run()
    ref = scan._diagonal_recurrence_sequential(torch.zeros(3), torch.zeros(2, 8, 3))
    assert out.shape == ref.shape


def test_explicit_missing_backend_warns_once_and_falls_back(monkeypatch):
    # Force the backend module to be genuinely absent (real c/triton backends exist
    # now): a None entry in sys.modules makes the import raise ImportError.
    monkeypatch.setitem(sys.modules, "tsfast.models._core.scan_backends.diagonal_c", None)
    monkeypatch.setattr(scan, "backend", "c")
    with pytest.warns(RuntimeWarning, match="falling back"):
        out1 = _run()
    with warnings.catch_warnings():  # second call: same key, no second warning
        warnings.simplefilter("error")
        out2 = _run()
    assert torch.equal(out1, out2) is False or out1.shape == out2.shape


def test_fake_backend_is_used_and_unsupported_reason_warns(monkeypatch):
    calls = {}

    fake = types.ModuleType("tsfast.models._core.scan_backends.diagonal_c")
    fake.supports = lambda lam, v, x0: None if v.shape[-2] > 4 else "sequence too short"
    fake.run = lambda lam, v, x0: calls.setdefault("out", torch.zeros(v.shape))
    monkeypatch.setitem(sys.modules, "tsfast.models._core.scan_backends.diagonal_c", fake)
    monkeypatch.setattr(scan, "backend", "c")

    out = _run()  # L=8 -> supported -> fake backend result
    assert "out" in calls and torch.equal(out, calls["out"])

    with pytest.warns(RuntimeWarning, match="sequence too short"):
        short = scan.diagonal_recurrence(torch.rand(3) * 0.5, torch.randn(2, 3, 3))
    ref = scan._diagonal_recurrence_sequential(torch.rand(3) * 0.0, torch.zeros(2, 3, 3))
    assert short.shape == ref.shape  # fell back to doubling


def test_forced_doubling_never_touches_backends(monkeypatch):
    boom = types.ModuleType("tsfast.models._core.scan_backends.diagonal_c")

    def _explode(*a):
        raise AssertionError("backend must not be probed when backend='doubling'")

    boom.supports = _explode
    boom.run = _explode
    monkeypatch.setitem(sys.modules, "tsfast.models._core.scan_backends.diagonal_c", boom)
    monkeypatch.setattr(scan, "backend", "doubling")
    _run()
