"""Regression tests verifying current behavior matches golden baselines.

These tests load the JSON fixtures produced by capture_baselines.py and
compare against live execution.
"""

import json
import math

import numpy as np
import pytest
import torch
from pathlib import Path

GOLDEN_DIR = Path(__file__).parent
DATA_GOLDEN = GOLDEN_DIR / "data_pipeline.json"
LEARNER_GOLDEN = GOLDEN_DIR / "learners.json"

PROJECT_ROOT = GOLDEN_DIR.parent.parent
WH_PATH = PROJECT_ROOT / "test_data" / "WienerHammerstein"
PINN_PATH = PROJECT_ROOT / "test_data" / "pinn"

SEED = 42

# Skip all tests if golden files don't exist yet
pytestmark = pytest.mark.skipif(
    not DATA_GOLDEN.exists() or not LEARNER_GOLDEN.exists(),
    reason="Golden baseline files not found. Run: python -m tests.golden.capture_baselines",
)


def set_deterministic():
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@pytest.fixture(scope="module")
def data_golden():
    with open(DATA_GOLDEN) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def learner_golden():
    with open(LEARNER_GOLDEN) as f:
        return json.load(f)


# --------------------------------------------------------------------------- #
#  Data pipeline configs (mirroring capture_baselines.py)
# --------------------------------------------------------------------------- #

DLS_CONFIGS = {
    "wh_simulation": dict(
        u=["u"], y=["y"], dataset=WH_PATH,
        win_sz=100, stp_sz=100, num_workers=0, n_batches_train=2,
    ),
    "pinn_simulation": dict(
        u=["u"], y=["x", "v"], dataset=PINN_PATH,
        win_sz=100, stp_sz=100, num_workers=0, n_batches_train=2,
    ),
}


# --------------------------------------------------------------------------- #
#  Data pipeline tests
# --------------------------------------------------------------------------- #

class TestDataPipelineGolden:
    @pytest.mark.parametrize("config_name", list(DLS_CONFIGS.keys()))
    def test_batch_shapes(self, data_golden, config_name):
        from tsfast.tsdata import create_dls

        golden = data_golden[config_name]
        dls = create_dls(**DLS_CONFIGS[config_name])
        batch = dls.one_batch()

        assert list(batch[0].shape) == golden["batch_xb_shape"]
        assert list(batch[1].shape) == golden["batch_yb_shape"]

    @pytest.mark.parametrize("config_name", list(DLS_CONFIGS.keys()))
    def test_dataloader_lengths(self, data_golden, config_name):
        from tsfast.tsdata import create_dls

        golden = data_golden[config_name]
        dls = create_dls(**DLS_CONFIGS[config_name])

        assert len(dls.train) == golden["train_len"]
        assert len(dls.valid) == golden["valid_len"]

    @pytest.mark.parametrize("config_name", list(DLS_CONFIGS.keys()))
    def test_norm_stats_structure(self, data_golden, config_name):
        """Verify norm stats have the right shape and nullness pattern.

        Values use a loose tolerance because estimate_norm_stats() samples
        shuffled training batches, so exact values vary across runs.
        """
        from tsfast.tsdata import create_dls

        golden = data_golden[config_name]
        dls = create_dls(**DLS_CONFIGS[config_name])
        norm_u, norm_y = dls.norm_stats

        # Feature counts must match exactly
        assert len(norm_u.mean) == len(golden["norm_u"]["mean"])
        assert len(norm_y.mean) == len(golden["norm_y"]["mean"])

        # Loose value check — stats are estimated from random batch samples
        np.testing.assert_allclose(norm_u.mean, golden["norm_u"]["mean"], atol=0.5)
        np.testing.assert_allclose(norm_y.mean, golden["norm_y"]["mean"], atol=0.5)



# --------------------------------------------------------------------------- #
#  Learner tests
# --------------------------------------------------------------------------- #

def _create_shared_dls():
    """Create shared DLS objects matching capture_baselines.py pattern."""
    from tsfast.tsdata import create_dls

    dls_sim = create_dls(u=["u"], y=["y"], dataset=WH_PATH,
                         win_sz=100, stp_sz=100, num_workers=0, n_batches_train=2)
    dls_pred = create_dls(u=["u"], y=["y"], dataset=WH_PATH,
                          win_sz=100, stp_sz=100, num_workers=0, n_batches_train=2)
    dls_pinn_pred = create_dls(u=["u"], y=["x", "v"], dataset=PINN_PATH,
                               win_sz=100, stp_sz=100, num_workers=0, n_batches_train=2)
    return dls_sim, dls_pred, dls_pinn_pred


# Module-level shared DLS — matches capture_baselines.py which creates DLS once
_shared_dls = None


def _get_shared_dls():
    global _shared_dls
    if _shared_dls is None:
        _shared_dls = _create_shared_dls()
    return _shared_dls


def _create_learner(name: str):
    """Recreate a learner matching the capture_baselines.py config."""
    dls_sim, dls_pred, dls_pinn_pred = _get_shared_dls()

    match name:
        case "RNNLearner":
            from tsfast.models.rnn import RNNLearner
            return RNNLearner(dls_sim, rnn_type="gru", hidden_size=20, num_layers=1)

        case "TCNLearner":
            from tsfast.models.cnn import TCNLearner
            return TCNLearner(dls_sim, num_layers=2, hidden_size=20)

        case "CRNNLearner":
            from tsfast.models.cnn import CRNNLearner
            return CRNNLearner(dls_sim, num_ft=10, num_cnn_layers=2, num_rnn_layers=1,
                               hs_cnn=10, hs_rnn=10)

        case "AR_RNNLearner":
            from tsfast.models.rnn import AR_RNNLearner
            return AR_RNNLearner(dls_sim, hidden_size=20, num_layers=1)

        case "FranSysLearner":
            from tsfast.prediction.fransys import FranSysLearner
            return FranSysLearner(dls_pred, init_sz=50, hidden_size=20, rnn_layer=1, attach_output=True)

        case "PIRNNLearner":
            from tsfast.pinn.pirnn import PIRNNLearner
            return PIRNNLearner(dls_pinn_pred, init_sz=20, hidden_size=20, rnn_layer=1,
                                attach_output=True)

        case _:
            raise ValueError(f"Unknown learner: {name}")


LEARNER_NAMES = [
    "RNNLearner", "TCNLearner", "CRNNLearner",
    "AR_RNNLearner", "FranSysLearner", "PIRNNLearner",
]


class TestLearnerGolden:
    @pytest.mark.parametrize("learner_name", LEARNER_NAMES)
    def test_param_count(self, learner_golden, learner_name):
        golden = learner_golden[learner_name]
        if "error" in golden:
            pytest.skip(f"Baseline capture failed: {golden['error']}")

        set_deterministic()
        lrn = _create_learner(learner_name)
        actual = sum(p.numel() for p in lrn.model.parameters())
        assert actual == golden["param_count"], (
            f"{learner_name} param count: {actual} != {golden['param_count']}"
        )

    @pytest.mark.slow
    @pytest.mark.parametrize("learner_name", LEARNER_NAMES)
    def test_val_loss(self, learner_golden, learner_name):
        golden = learner_golden[learner_name]
        if "error" in golden:
            pytest.skip(f"Baseline capture failed: {golden['error']}")

        set_deterministic()
        lrn = _create_learner(learner_name)
        lrn.fit(1, 3e-3)
        val_loss = float(lrn.recorder.values[-1][1])

        assert not math.isnan(val_loss), f"{learner_name} produced NaN val_loss"
        assert val_loss == pytest.approx(golden["val_loss_1epoch"], rel=0.05), (
            f"{learner_name} val_loss: {val_loss:.6f} != {golden['val_loss_1epoch']:.6f}"
        )
