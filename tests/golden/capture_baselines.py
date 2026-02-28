"""Capture golden baselines for migration regression testing.

Run from the project root:
    python -m tests.golden.capture_baselines

Produces:
    tests/golden/data_pipeline.json  — normalization stats, batch shapes, window counts
    tests/golden/learners.json       — per-learner validation loss, param counts, output shapes
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

SEED = 42
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
GOLDEN_DIR = PROJECT_ROOT / "tests" / "golden"
WH_PATH = PROJECT_ROOT / "test_data" / "WienerHammerstein"
PINN_PATH = PROJECT_ROOT / "test_data" / "pinn"


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        return super().default(obj)


def set_deterministic():
    """Set all seeds for reproducibility."""
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _norm_pair_to_dict(np_obj) -> dict | None:
    """Convert a NormPair to a JSON-serializable dict."""
    if np_obj is None:
        return None
    return {
        "mean": np_obj.mean,
        "std": np_obj.std,
        "min": np_obj.min,
        "max": np_obj.max,
    }


# --------------------------------------------------------------------------- #
#  Data pipeline baselines
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


def capture_data_pipeline() -> dict:
    """Record create_dls() outputs for all configs."""
    from tsfast.tsdata import create_dls

    results = {}
    for name, kwargs in DLS_CONFIGS.items():
        dls = create_dls(**kwargs)
        batch = dls.one_batch()
        norm_u, norm_y = dls.norm_stats

        results[name] = {
            "train_len": len(dls.train),
            "valid_len": len(dls.valid),
            "batch_xb_shape": list(batch[0].shape),
            "batch_yb_shape": list(batch[1].shape),
            "norm_u": _norm_pair_to_dict(norm_u),
            "norm_y": _norm_pair_to_dict(norm_y),
        }
        print(f"  [{name}] xb={batch[0].shape}, yb={batch[1].shape}, "
              f"train={len(dls.train)}, valid={len(dls.valid)}")

    return results


# --------------------------------------------------------------------------- #
#  Learner baselines
# --------------------------------------------------------------------------- #

def _param_count(model) -> int:
    return sum(p.numel() for p in model.parameters())


def _capture_learner(name: str, create_fn) -> dict:
    """Create a learner, train 1 epoch, record outputs."""
    set_deterministic()
    lrn = create_fn()

    lrn.fit(1, 3e-3)
    val_loss = float(lrn.recorder.values[-1][1])

    # Forward pass for output shape
    batch = next(iter(lrn.dls.valid))
    lrn.model.eval()
    with torch.no_grad():
        pred = lrn.model(batch[0].to(next(lrn.model.parameters()).device))
    # Some models return (output, state) tuple
    if isinstance(pred, tuple):
        pred = pred[0]

    entry = {
        "val_loss_1epoch": val_loss,
        "param_count": _param_count(lrn.model),
        "output_shape": list(pred.shape),
    }
    print(f"  [{name}] val_loss={val_loss:.6f}, params={entry['param_count']}, "
          f"output={list(pred.shape)}")
    return entry


def capture_learners() -> dict:
    """Train each learner type for 1 epoch and record validation loss."""
    from tsfast.tsdata import create_dls
    from tsfast.models.rnn import RNNLearner
    from tsfast.models.cnn import TCNLearner, CRNNLearner
    from tsfast.prediction.fransys import FranSysLearner
    from tsfast.pinn.pirnn import PIRNNLearner

    # Build DataLoaders — reuse across learners of same type
    dls_sim = create_dls(
        u=["u"], y=["y"], dataset=WH_PATH,
        win_sz=100, stp_sz=100, num_workers=0, n_batches_train=2,
    )
    dls_pred = create_dls(
        u=["u"], y=["y"], dataset=WH_PATH,
        win_sz=100, stp_sz=100, num_workers=0, n_batches_train=2,
    )
    dls_pinn_pred = create_dls(
        u=["u"], y=["x", "v"], dataset=PINN_PATH,
        win_sz=100, stp_sz=100, num_workers=0, n_batches_train=2,
    )

    learner_configs = {
        "RNNLearner": lambda: RNNLearner(
            dls_sim, rnn_type="gru", hidden_size=20, num_layers=1,
        ),
        "TCNLearner": lambda: TCNLearner(
            dls_sim, num_layers=2, hidden_size=20,
        ),
        "CRNNLearner": lambda: CRNNLearner(
            dls_sim, num_ft=10, num_cnn_layers=2, num_rnn_layers=1,
            hs_cnn=10, hs_rnn=10,
        ),
        "AR_RNNLearner": lambda: _create_ar_rnn(dls_sim),
        "FranSysLearner": lambda: FranSysLearner(
            dls_pred, init_sz=50, hidden_size=20, rnn_layer=1,
        ),
        "PIRNNLearner": lambda: PIRNNLearner(
            dls_pinn_pred, init_sz=20, hidden_size=20, rnn_layer=1,
            attach_output=True,
        ),
    }

    results = {}
    for name, create_fn in learner_configs.items():
        try:
            results[name] = _capture_learner(name, create_fn)
        except Exception as e:
            print(f"  [{name}] FAILED: {e}")
            results[name] = {"error": str(e)}

    return results


def _create_ar_rnn(dls):
    """Create AR_RNNLearner — import separately to avoid circular issues."""
    from tsfast.models.rnn import AR_RNNLearner
    return AR_RNNLearner(dls, hidden_size=20, num_layers=1)


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #

def main():
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)

    print("Capturing data pipeline baselines...")
    data_results = capture_data_pipeline()
    data_path = GOLDEN_DIR / "data_pipeline.json"
    with open(data_path, "w") as f:
        json.dump(data_results, f, indent=2, cls=NumpyEncoder)
    print(f"  Saved to {data_path}\n")

    print("Capturing learner baselines...")
    learner_results = capture_learners()
    learner_path = GOLDEN_DIR / "learners.json"
    with open(learner_path, "w") as f:
        json.dump(learner_results, f, indent=2, cls=NumpyEncoder)
    print(f"  Saved to {learner_path}\n")

    # Summary
    n_ok = sum(1 for v in learner_results.values() if "error" not in v)
    n_err = sum(1 for v in learner_results.values() if "error" in v)
    print(f"Done. {n_ok} learners captured, {n_err} failed.")
    if n_err:
        sys.exit(1)


if __name__ == "__main__":
    main()
