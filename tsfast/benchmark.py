__all__ = ["BENCHMARK_DL_KWARGS", "create_dls_from_spec"]

from fastai.data.all import *
from .datasets.core import create_dls_downl
import identibench as idb

BENCHMARK_DL_KWARGS = {
    # Simulation Benchmarks
    "BenchmarkWH_Simulation": {"win_sz": 200},
    "BenchmarkSilverbox_Simulation": {"win_sz": 200},
    "BenchmarkCascadedTanks_Simulation": {"win_sz": 150, "bs": 16},
    "BenchmarkEMPS_Simulation": {"win_sz": 1000},
    "BenchmarkCED_Simulation": {"win_sz": 100, "bs": 16},
    "BenchmarkNoisyWH_Simulation": {"win_sz": 100, "stp_sz": 50},
    "BenchmarkRobotForward_Simulation": {"win_sz": 300, "valid_stp_sz": 4},
    "BenchmarkRobotInverse_Simulation": {"win_sz": 300, "valid_stp_sz": 4},
    "BenchmarkShip_Simulation": {"win_sz": 100},
    "BenchmarkQuadPelican_Simulation": {"win_sz": 300, "valid_stp_sz": 40},
    "BenchmarkQuadPi_Simulation": {"win_sz": 200, "valid_stp_sz": 20},
    # Prediction Benchmarks
    "BenchmarkWH_Prediction": {},
    "BenchmarkSilverbox_Prediction": {},
    "BenchmarkCascadedTanks_Prediction": {"bs": 16},
    "BenchmarkEMPS_Prediction": {},
    "BenchmarkCED_Prediction": {"bs": 16},
    "BenchmarkNoisyWH_Prediction": {"stp_sz": 50},
    "BenchmarkRobotForward_Prediction": {"valid_stp_sz": 4},
    "BenchmarkRobotInverse_Prediction": {"valid_stp_sz": 4},
    "BenchmarkShip_Prediction": {},
    "BenchmarkQuadPelican_Prediction": {"valid_stp_sz": 40},
    "BenchmarkQuadPi_Prediction": {"valid_stp_sz": 20},
}


@delegates(create_dls_downl, keep=True)
def create_dls_from_spec(
    spec: idb.benchmark.BenchmarkSpecBase,  # Specification of the benchmark from identibench
    **kwargs,  # kwargs for create_dls_downl
):
    """
    Create a dataloaders object from identibench benchmark specification. Extracts
    benchmark specific kwargs from BENCHMARK_DL_KWARGS and adds them to the kwargs for create_dls_downl.
    """
    # add kwargs form spec to dl_kwargs if the key is not already in dl_kwargs
    spec_kwargs = {
        "u": spec.u_cols,
        "y": spec.y_cols,
        "download_function": spec.download_func,
        "dataset": spec.dataset_path,
    }

    # add prediction specific kwargs
    if isinstance(spec, idb.benchmark.BenchmarkSpecPrediction):
        spec_kwargs.update(
            {"win_sz": spec.pred_horizon + spec.init_window, "valid_stp_sz": spec.pred_step, "prediction": True}
        )

    # add tsfast specific kwargs using BENCHMARK_DL_KWARGS
    if spec.name in BENCHMARK_DL_KWARGS:
        spec_kwargs.update(BENCHMARK_DL_KWARGS[spec.name])

    dl_kwargs = {**spec_kwargs, **kwargs}
    return create_dls_downl(**dl_kwargs)
