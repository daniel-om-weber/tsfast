"""Tests for tsfast.training.profiling — dataloader profiling plugin."""

import time

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from tsfast.training import Learner
from tsfast.training.profiling import DataProfiler, _TimedIterator


# ──────────────────────────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────────────────────────


class _SyntheticDls:
    """Minimal DataLoaders-like object for unit tests."""

    def __init__(self, n_u=1, n_y=1, seq_len=100, n_train=16, n_valid=8, bs=4):
        x_train = torch.randn(n_train, seq_len, n_u)
        y_train = torch.randn(n_train, seq_len, n_y)
        x_valid = torch.randn(n_valid, seq_len, n_u)
        y_valid = torch.randn(n_valid, seq_len, n_y)
        self.train = DataLoader(TensorDataset(x_train, y_train), batch_size=bs, shuffle=True)
        self.valid = DataLoader(TensorDataset(x_valid, y_valid), batch_size=bs)
        self.test = None


class _SlowDataLoader:
    """DataLoader that sleeps before yielding each batch."""

    def __init__(self, dl, delay: float):
        self._dl = dl
        self._delay = delay

    def __iter__(self):
        for batch in self._dl:
            time.sleep(self._delay)
            yield batch

    def __len__(self):
        return len(self._dl)

    def __getattr__(self, name):
        return getattr(self._dl, name)


class _SimpleModel(nn.Module):
    def __init__(self, n_u=1, n_y=1):
        super().__init__()
        self.fc = nn.Linear(n_u, n_y)

    def forward(self, x):
        return self.fc(x)


def _make_learner(**dls_kwargs):
    dls = _SyntheticDls(**dls_kwargs)
    model = _SimpleModel(n_u=dls_kwargs.get("n_u", 1), n_y=dls_kwargs.get("n_y", 1))
    return Learner(model, dls, loss_func=nn.MSELoss(), device=torch.device("cpu"))


# ──────────────────────────────────────────────────────────────────────────────
#  _TimedIterator
# ──────────────────────────────────────────────────────────────────────────────


class TestTimedIterator:
    def test_len_delegation(self):
        dls = _SyntheticDls()
        prof = DataProfiler()
        timed = _TimedIterator(dls.train, prof)
        assert len(timed) == len(dls.train)

    def test_getattr_delegation(self):
        dls = _SyntheticDls()
        prof = DataProfiler()
        timed = _TimedIterator(dls.train, prof)
        assert timed.dataset is dls.train.dataset

    def test_records_data_times(self):
        dls = _SyntheticDls(n_train=8, bs=4)
        prof = DataProfiler()
        timed = _TimedIterator(dls.train, prof)
        batches = list(timed)
        assert len(batches) == 2  # 8 samples / 4 batch_size
        assert len(prof.data_times) == 2
        assert all(t >= 0 for t in prof.data_times)

    def test_reusable_across_epochs(self):
        dls = _SyntheticDls(n_train=8, bs=4)
        prof = DataProfiler()
        timed = _TimedIterator(dls.train, prof)
        list(timed)  # epoch 1
        list(timed)  # epoch 2
        assert len(prof.data_times) == 4  # 2 batches * 2 epochs


# ──────────────────────────────────────────────────────────────────────────────
#  DataProfiler
# ──────────────────────────────────────────────────────────────────────────────


class TestDataProfiler:
    def test_profile_data_records_times(self, capsys):
        lrn = _make_learner()
        n_batches = len(lrn.dls.train)
        with lrn.no_bar():
            with DataProfiler.profile(lrn) as prof:
                lrn.fit(2)

        assert len(prof.data_times) == n_batches * 2
        assert len(prof.step_times) == n_batches * 2
        # Summary was printed
        captured = capsys.readouterr()
        assert "DataLoader Profile" in captured.out

    def test_restores_state_after_exit(self, capsys):
        lrn = _make_learner()
        original_dl = lrn.dls.train

        with lrn.no_bar():
            with DataProfiler.profile(lrn):
                assert lrn.dls.train is not original_dl
                lrn.fit(1)

        assert lrn.dls.train is original_dl
        # Instance attribute removed — falls back to class method
        assert "_train_one_batch" not in lrn.__dict__

    def test_restores_on_exception(self, capsys):
        lrn = _make_learner()
        original_dl = lrn.dls.train

        class _Boom(Exception):
            pass

        try:
            with DataProfiler.profile(lrn):
                raise _Boom()
        except _Boom:
            pass

        assert lrn.dls.train is original_dl

    def test_stall_detection(self, capsys):
        lrn = _make_learner(n_train=8, bs=4)
        # Replace train loader with a slow one
        lrn.dls.train = _SlowDataLoader(lrn.dls.train, delay=0.02)

        with lrn.no_bar():
            with DataProfiler.profile(lrn, stall_threshold=0.01) as prof:
                lrn.fit(1)

        assert prof.n_stalls > 0
        assert prof.stall_pct > 0

    def test_no_stalls_with_fast_loader(self, capsys):
        lrn = _make_learner()
        with lrn.no_bar():
            with DataProfiler.profile(lrn, stall_threshold=1.0) as prof:
                lrn.fit(1)

        assert prof.n_stalls == 0

    def test_summary_format(self):
        prof = DataProfiler(stall_threshold=0.005)
        prof.data_times = [0.001, 0.002, 0.001, 0.010]
        prof.step_times = [0.05, 0.04, 0.05, 0.06]
        s = prof.summary()
        assert "Data loading:" in s
        assert "Training step:" in s
        assert "Verdict:" in s
        assert "Stalls" in s

    def test_summary_empty(self):
        prof = DataProfiler()
        assert "No batches recorded" in prof.summary()

    def test_data_wait_fraction(self):
        prof = DataProfiler()
        prof.data_times = [0.5, 0.5]
        prof.step_times = [0.5, 0.5]
        assert abs(prof.data_wait_fraction - 0.5) < 1e-6

    def test_summary_bottleneck_verdict(self):
        prof = DataProfiler()
        # Data takes way more time than step
        prof.data_times = [0.1] * 10
        prof.step_times = [0.01] * 10
        s = prof.summary()
        assert "bottleneck" in s.lower()
