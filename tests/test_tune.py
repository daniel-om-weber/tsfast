"""Tests for the Ray Tune hyperparameter optimization integration."""

import os
import pytest

ray = pytest.importorskip("ray")
tune = pytest.importorskip("ray.tune")


@pytest.fixture(scope="module")
def ray_init_shutdown():
    """Start and stop Ray once for the module."""
    os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"
    ray.init(num_cpus=2, num_gpus=0, include_dashboard=False)
    yield
    ray.shutdown()


@pytest.fixture(scope="module")
def dls_silverbox():
    """Small Silverbox DataLoaders for tune tests."""
    from tsfast.tsdata.benchmark import create_dls_silverbox

    return create_dls_silverbox(bs=16, win_sz=50, n_batches_train=2, n_batches_valid=2)


@pytest.mark.slow
def test_trainable_cpu(ray_init_shutdown, dls_silverbox):
    """Training function with ray_device completes on CPU-only workers.

    Regression test: DataLoaders created on a CUDA host keep device='cuda'
    after being serialized to Ray workers that only have CPU.  The trainable
    must work correctly on CPU-only workers.
    """
    from tsfast.training import RNNLearner, fun_rmse
    from tsfast.tune import ray_device, report_metrics, resume_checkpoint

    dls = dls_silverbox

    def trainable(config):
        lrn = RNNLearner(
            dls,
            rnn_type=config["rnn_type"],
            hidden_size=config["hidden_size"],
            n_skip=5,
            metrics=[fun_rmse],
            device=ray_device(),
        )
        resume_checkpoint(lrn)
        with lrn.no_bar(), report_metrics(lrn):
            lrn.fit_flat_cos(config["n_epoch"], lr=config.get("lr"))

    tuner = tune.Tuner(
        tune.with_resources(trainable, {"cpu": 1, "gpu": 0}),
        param_space={
            "rnn_type": tune.choice(["gru"]),
            "hidden_size": tune.choice([32]),
            "n_epoch": 1,
            "lr": 3e-3,
        },
        tune_config=tune.TuneConfig(metric="valid_loss", mode="min", num_samples=1),
    )
    results = tuner.fit()

    assert not results.errors
    best = results.get_best_result()
    assert best.config["rnn_type"] == "gru"
    assert best.config["hidden_size"] == 32
    assert "valid_loss" in best.metrics


@pytest.mark.slow
def test_trainable_loguniform(ray_init_shutdown, dls_silverbox):
    """tune.loguniform produces a concrete float by the time the trainable runs."""
    from tsfast.training import RNNLearner, fun_rmse
    from tsfast.tune import ray_device, report_metrics

    dls = dls_silverbox

    def trainable(config):
        lrn = RNNLearner(
            dls,
            rnn_type=config["rnn_type"],
            hidden_size=config["hidden_size"],
            n_skip=5,
            metrics=[fun_rmse],
            device=ray_device(),
        )
        with lrn.no_bar(), report_metrics(lrn):
            lrn.fit_flat_cos(config["n_epoch"], lr=config.get("lr"))

    tuner = tune.Tuner(
        tune.with_resources(trainable, {"cpu": 1, "gpu": 0}),
        param_space={
            "rnn_type": tune.choice(["gru"]),
            "hidden_size": tune.choice([32]),
            "lr": tune.loguniform(1e-4, 1e-2),
            "n_epoch": 1,
        },
        tune_config=tune.TuneConfig(metric="valid_loss", mode="min", num_samples=1),
    )
    results = tuner.fit()

    assert not results.errors
    best = results.get_best_result()
    assert isinstance(best.config["lr"], float)
    assert 1e-4 <= best.config["lr"] <= 1e-2


@pytest.mark.slow
def test_report_metrics_restores_log_epoch(ray_init_shutdown, dls_silverbox):
    """report_metrics restores the original log_epoch after the context exits."""
    from tsfast.training import RNNLearner, fun_rmse
    from tsfast.tune import report_metrics

    lrn = RNNLearner(
        dls_silverbox,
        rnn_type="gru",
        hidden_size=16,
        n_skip=5,
        metrics=[fun_rmse],
    )

    assert "log_epoch" not in lrn.__dict__

    with report_metrics(lrn):
        assert "log_epoch" in lrn.__dict__

    assert "log_epoch" not in lrn.__dict__


@pytest.mark.slow
def test_checkpoint_every(ray_init_shutdown, dls_silverbox):
    """checkpoint_every controls how often checkpoints are saved."""
    from unittest.mock import patch, MagicMock
    from tsfast.training import RNNLearner, fun_rmse
    from tsfast.tune import report_metrics

    lrn = RNNLearner(
        dls_silverbox, rnn_type="gru", hidden_size=16, n_skip=5, metrics=[fun_rmse],
    )

    calls = []
    orig_report = ray.tune.report

    def tracking_report(metrics, *, checkpoint=None):
        calls.append(checkpoint is not None)
        return orig_report(metrics, checkpoint=checkpoint)

    with patch("ray.tune.report", side_effect=tracking_report):
        with lrn.no_bar(), report_metrics(lrn, checkpoint_every=2):
            lrn.fit_flat_cos(4, lr=3e-3)

    assert calls == [False, True, False, True]


@pytest.mark.slow
def test_checkpoint_every_none(ray_init_shutdown, dls_silverbox):
    """checkpoint_every=None disables checkpointing entirely."""
    from unittest.mock import patch
    from tsfast.training import RNNLearner, fun_rmse
    from tsfast.tune import report_metrics

    lrn = RNNLearner(
        dls_silverbox, rnn_type="gru", hidden_size=16, n_skip=5, metrics=[fun_rmse],
    )

    calls = []
    orig_report = ray.tune.report

    def tracking_report(metrics, *, checkpoint=None):
        calls.append(checkpoint is not None)
        return orig_report(metrics, checkpoint=checkpoint)

    with patch("ray.tune.report", side_effect=tracking_report):
        with lrn.no_bar(), report_metrics(lrn, checkpoint_every=None):
            lrn.fit_flat_cos(3, lr=3e-3)

    assert calls == [False, False, False]


def test_learner_freed_by_refcount(dls_silverbox):
    """Learner is freed by refcounting alone after fit() — no gc.collect() needed."""
    import gc
    import weakref
    from tsfast.training import RNNLearner, fun_rmse

    lrn = RNNLearner(
        dls_silverbox, rnn_type="gru", hidden_size=16, n_skip=5, metrics=[fun_rmse],
    )
    ref = weakref.ref(lrn)
    lrn.fit_flat_cos(1, lr=3e-3)

    gc.disable()
    try:
        del lrn
        assert ref() is None, "Learner not freed by refcount — cycle exists after fit()"
    finally:
        gc.enable()
