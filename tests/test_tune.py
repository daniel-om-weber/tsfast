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


@pytest.mark.slow
def test_learner_trainable_basic(ray_init_shutdown, dls_silverbox):
    """LearnerTrainable runs training iterations and reports metrics."""
    from tsfast.training import RNNLearner, fun_rmse
    from tsfast.tune import LearnerTrainable, ray_device

    dls = dls_silverbox

    def create_learner(config):
        return RNNLearner(
            dls,
            rnn_type="gru",
            hidden_size=config["hidden_size"],
            n_skip=5,
            metrics=[fun_rmse],
            device=ray_device(),
        )

    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(LearnerTrainable, create_learner=create_learner),
            {"cpu": 1, "gpu": 0},
        ),
        param_space={"hidden_size": 16, "lr": 3e-3},
        run_config=tune.RunConfig(stop={"training_iteration": 2}),
        tune_config=tune.TuneConfig(metric="valid_loss", mode="min", num_samples=1),
    )
    results = tuner.fit()

    assert not results.errors
    best = results.get_best_result()
    assert "valid_loss" in best.metrics
    assert "train_loss" in best.metrics
    assert "fun_rmse" in best.metrics


@pytest.mark.slow
def test_learner_trainable_reset_config(ray_init_shutdown, dls_silverbox):
    """reset_config uses apply_config when provided, returns False otherwise."""
    from tsfast.training import RNNLearner, fun_rmse
    from tsfast.tune import LearnerTrainable

    dls = dls_silverbox

    def create_learner(config):
        return RNNLearner(
            dls, rnn_type="gru", hidden_size=16, n_skip=5, metrics=[fun_rmse],
        )

    def apply_config(lrn, config):
        for pg in lrn.opt.param_groups:
            pg["lr"] = config["lr"]

    trainable_with = tune.with_parameters(
        LearnerTrainable, create_learner=create_learner, apply_config=apply_config,
    )
    t = trainable_with({"lr": 1e-3})
    assert t.reset_config({"lr": 5e-4}) is True
    assert t.lrn.opt.param_groups[0]["lr"] == 5e-4
    t.cleanup()

    trainable_without = tune.with_parameters(
        LearnerTrainable, create_learner=create_learner,
    )
    t2 = trainable_without({"lr": 1e-3})
    assert t2.reset_config({"lr": 5e-4}) is False
    t2.cleanup()


@pytest.mark.slow
def test_learner_trainable_scheduler(ray_init_shutdown, dls_silverbox):
    """LearnerTrainable with scheduler_fn applies LR schedule across iterations."""
    from torch.optim.lr_scheduler import LambdaLR
    from tsfast.training import RNNLearner, fun_rmse
    from tsfast.training.schedulers import sched_flat_cos
    from tsfast.tune import LearnerTrainable, ray_device

    dls = dls_silverbox

    def create_learner(config):
        return RNNLearner(
            dls, rnn_type="gru", hidden_size=16, n_skip=5,
            metrics=[fun_rmse], device=ray_device(),
        )

    def flat_cos_fn(opt, total_steps):
        return LambdaLR(opt, lambda s: sched_flat_cos(s / total_steps))

    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(
                LearnerTrainable,
                create_learner=create_learner,
                scheduler_fn=flat_cos_fn,
            ),
            {"cpu": 1, "gpu": 0},
        ),
        param_space={"hidden_size": 16, "lr": 3e-3, "n_iter": 3},
        run_config=tune.RunConfig(stop={"training_iteration": 3}),
        tune_config=tune.TuneConfig(metric="valid_loss", mode="min", num_samples=1),
    )
    results = tuner.fit()

    assert not results.errors
    best = results.get_best_result()
    assert "valid_loss" in best.metrics


@pytest.mark.slow
def test_learner_trainable_scheduler_checkpoint(ray_init_shutdown, dls_silverbox):
    """Scheduler state is preserved across checkpoint save/restore."""
    import tempfile
    from torch.optim.lr_scheduler import LambdaLR
    from tsfast.training import RNNLearner, fun_rmse
    from tsfast.training.schedulers import sched_flat_cos
    from tsfast.tune import LearnerTrainable

    dls = dls_silverbox

    def create_learner(config):
        return RNNLearner(
            dls, rnn_type="gru", hidden_size=16, n_skip=5, metrics=[fun_rmse],
        )

    def flat_cos_fn(opt, total_steps):
        return LambdaLR(opt, lambda s: sched_flat_cos(s / total_steps))

    trainable = tune.with_parameters(
        LearnerTrainable,
        create_learner=create_learner,
        scheduler_fn=flat_cos_fn,
    )({"lr": 3e-3, "n_iter": 4})

    trainable.step()
    trainable.step()
    last_epoch_before = trainable.lrn.sched.last_epoch

    with tempfile.TemporaryDirectory() as d:
        trainable.save_checkpoint(d)
        trainable.load_checkpoint(d)

    assert trainable.lrn.sched.last_epoch == last_epoch_before
    trainable.cleanup()


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
