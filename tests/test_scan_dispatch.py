"""Tests for backend preference and resolution: config scoping, fallback chain, once-per-process warning."""

import sys
import types
import warnings

import pytest
import torch

import tsfast.models._core.dispatch as dispatch
import tsfast.models._core.scan as scan
from tsfast.models import get_backend, set_backend, use_backend

_C_PATH = "tsfast.models._core.scan_backends.diagonal_c"


@pytest.fixture(autouse=True)
def _reset_state(monkeypatch):
    monkeypatch.setattr(dispatch, "_default_backend", "auto")
    monkeypatch.setattr(dispatch, "_warned", set())
    yield


def _run(lam=None, v=None):
    lam = torch.rand(3) * 0.5 if lam is None else lam
    v = torch.randn(2, 8, 3) if v is None else v
    return scan.diagonal_recurrence(lam, v)


def test_use_backend_scoping_and_set_backend():
    assert get_backend() == "auto"
    set_backend("reference")
    assert get_backend() == "reference"
    with use_backend("c"):
        assert get_backend() == "c"
        with use_backend("auto"):
            assert get_backend() == "auto"
        assert get_backend() == "c"
    assert get_backend() == "reference"


def test_unknown_backend_name_rejected():
    with pytest.raises(ValueError, match="unknown backend"):
        set_backend("doubling")
    with pytest.raises(ValueError, match="unknown backend"):
        with use_backend("trtion"):
            pass


def test_family_missing_for_op_is_silent():
    # "metal" is a valid family but no scan backend exists for it: reference path, no warning.
    with use_backend("metal"):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            out = _run()
    ref = scan._diagonal_recurrence_sequential(torch.zeros(3), torch.zeros(2, 8, 3))
    assert out.shape == ref.shape


def test_import_failure_warns_once_and_falls_back(monkeypatch):
    # A None entry in sys.modules makes the import raise ImportError.
    monkeypatch.setitem(sys.modules, _C_PATH, None)
    with use_backend("c"):
        with pytest.warns(RuntimeWarning, match="reference implementation"):
            out1 = _run()
        with warnings.catch_warnings():  # second call: same key, no second warning
            warnings.simplefilter("error")
            out2 = _run()
    assert out1.shape == out2.shape


def test_fake_backend_is_used_and_unsupported_reason_warns(monkeypatch):
    calls = {}

    fake = types.ModuleType(_C_PATH)
    fake.supports = lambda lam, v, x0: None if v.shape[-2] > 4 else "sequence too short"
    fake.forward = lambda lam, v, x0: calls.setdefault("out", torch.zeros(v.shape))
    monkeypatch.setitem(sys.modules, _C_PATH, fake)

    with use_backend("c"):
        out = _run()  # L=8 -> supported -> fake backend result
        assert "out" in calls and torch.equal(out.reshape(calls["out"].shape), calls["out"])

        with pytest.warns(RuntimeWarning, match="sequence too short"):
            short = scan.diagonal_recurrence(torch.rand(3) * 0.5, torch.randn(2, 3, 3))
    assert short.shape == (2, 3, 3)  # fell back to doubling


def test_reference_never_touches_backends(monkeypatch):
    boom = types.ModuleType(_C_PATH)

    def _explode(*a):
        raise AssertionError("backend must not be probed under the reference preference")

    boom.supports = _explode
    boom.forward = _explode
    monkeypatch.setitem(sys.modules, _C_PATH, boom)
    with use_backend("reference"):
        _run()


def test_scan_op_compiles_fullgraph():
    def f(lam, v):
        return scan.diagonal_recurrence(lam, v).sum()

    lam = (torch.rand(4) * 0.5).requires_grad_()
    v = torch.randn(2, 16, 4, requires_grad=True)
    eager = f(lam, v)
    out = torch.compile(f, fullgraph=True)(lam, v)
    out.backward()
    assert torch.allclose(out, eager, rtol=1e-5, atol=1e-6)
    assert lam.grad is not None and v.grad is not None
