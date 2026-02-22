"""Tests for the Ray Tune hyperparameter optimization integration."""

import pytest
import torch

ray = pytest.importorskip("ray")
tune = pytest.importorskip("ray.tune")


@pytest.fixture(scope="module", autouse=True)
def ray_init_shutdown():
    """Start and stop Ray once for the module."""
    ray.init(num_cpus=2, num_gpus=0, log_to_driver=True)
    yield
    ray.shutdown()


@pytest.fixture(scope="module")
def dls_silverbox():
    """Small Silverbox DataLoaders for tune tests."""
    from tsfast.datasets.benchmark import create_dls_silverbox

    return create_dls_silverbox(bs=16, win_sz=500, stp_sz=10)


@pytest.mark.slow
def test_learner_optimize_cpu_only(dls_silverbox):
    """HPOptimizer.optimize completes when workers have no GPU access.

    Regression test: DataLoaders created on a CUDA host keep device='cuda'
    after being serialized to Ray workers that only have CPU.  The optimize
    path must move the DataLoaders to the worker's device before building
    the Learner, otherwise ``dls.one_batch()`` raises
    ``RuntimeError: No CUDA GPUs are available``.
    """
    from tsfast.models.rnn import RNNLearner
    from tsfast.tune import HPOptimizer
    from tsfast.learner.losses import fun_rmse

    def create_learner(dls, config):
        return RNNLearner(
            dls,
            rnn_type=config["rnn_type"],
            hidden_size=config["hidden_size"],
            n_skip=50,
            metrics=[fun_rmse],
        )

    search_config = {
        "rnn_type": tune.choice(["gru"]),
        "hidden_size": tune.choice([32]),
        "n_epoch": 1,
        "lr": 3e-3,
    }

    optimizer = HPOptimizer(create_lrn=create_learner, dls=dls_silverbox)
    results = optimizer.optimize(
        config=search_config,
        num_samples=1,
        resources_per_trial={"cpu": 1, "gpu": 0},
    )

    assert results is not None
    best = optimizer.analysis.get_best_config(metric="valid_loss", mode="min")
    assert "rnn_type" in best
    assert "hidden_size" in best


@pytest.mark.slow
def test_learner_optimize_callable_lr(dls_silverbox):
    """HPOptimizer.optimize works when lr is a callable sampler.

    Regression test: ``log_uniform`` returns a plain callable, not a Ray Tune
    sampling primitive.  ``learner_optimize`` must call it to obtain a float
    before assigning to ``lrn.lr``, otherwise ``fit_flat_cos`` raises
    ``TypeError: unsupported operand type(s) for /: 'function' and 'float'``.
    """
    from tsfast.models.rnn import RNNLearner
    from tsfast.tune import HPOptimizer, log_uniform
    from tsfast.learner.losses import fun_rmse

    def create_learner(dls, config):
        return RNNLearner(
            dls,
            rnn_type=config["rnn_type"],
            hidden_size=config["hidden_size"],
            n_skip=50,
            metrics=[fun_rmse],
        )

    search_config = {
        "rnn_type": tune.choice(["gru"]),
        "hidden_size": tune.choice([32]),
        "lr": log_uniform(1e-4, 1e-2),
        "n_epoch": 1,
    }

    optimizer = HPOptimizer(create_lrn=create_learner, dls=dls_silverbox)
    results = optimizer.optimize(
        config=search_config,
        num_samples=1,
        resources_per_trial={"cpu": 1, "gpu": 0},
    )

    assert results is not None
    best = optimizer.analysis.get_best_config(metric="valid_loss", mode="min")
    assert "rnn_type" in best
    assert isinstance(best["lr"], float) or callable(best["lr"])
