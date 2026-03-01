"""Pre-configured DataLoader factories for identibench benchmark datasets."""

import os
from functools import partial
from pathlib import Path

import identibench as idb

from .pipeline import DataLoaders, create_dls
from .split import is_dataset_directory

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
    # Prediction Benchmarks (Phase 1: simulation-only, prediction deferred to Phase 3)
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


def _get_default_dataset_path() -> Path:
    """Return default directory for storing datasets."""
    data_dir = Path.home() / ".tsfast" / "datasets"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def _get_dataset_path() -> Path:
    """Return dataset directory from TSFAST_PATH env var, or the default."""
    env_path = os.getenv("TSFAST_PATH")
    return Path(env_path) if env_path else _get_default_dataset_path()


def create_dls_downl(
    u: list[str],
    y: list[str],
    dataset: Path | str | None = None,
    download_function=None,
    **kwargs,
) -> DataLoaders:
    """Create DataLoaders, downloading the dataset first if needed.

    Args:
        u: list of input signal names
        y: list of output signal names
        dataset: path to the dataset directory
        download_function: callable that downloads the dataset to a given path
    """
    if dataset is None and download_function is not None:
        dataset = _get_dataset_path() / download_function.__name__
    elif dataset is not None:
        dataset = Path(dataset)
    else:
        raise ValueError("Must provide either dataset path or download_function")

    if not is_dataset_directory(dataset):
        if download_function is not None:
            print(f'Dataset not found. Downloading it to "{dataset}"')
            download_function(dataset)
        else:
            raise ValueError(f"{dataset} does not contain a dataset. Check the path or activate the download flag.")

    return create_dls(u=u, y=y, dataset=dataset, **kwargs)


def create_dls_from_spec(
    spec: idb.benchmark.BenchmarkSpecBase,
    **kwargs,
) -> DataLoaders:
    """Create DataLoaders from an identibench benchmark specification.

    Args:
        spec: benchmark specification from identibench
    """
    spec_kwargs = {
        "u": spec.u_cols,
        "y": spec.y_cols,
        "download_function": spec.download_func,
        "dataset": spec.dataset_path,
    }

    # Prediction specs: in Phase 1 we only handle the window sizing, not the
    # actual prediction input concatenation (deferred to Phase 3).
    if isinstance(spec, idb.benchmark.BenchmarkSpecPrediction):
        spec_kwargs.update(
            {
                "win_sz": spec.pred_horizon + spec.init_window,
                "valid_stp_sz": spec.pred_step,
            }
        )

    if spec.name in BENCHMARK_DL_KWARGS:
        spec_kwargs.update(BENCHMARK_DL_KWARGS[spec.name])

    dl_kwargs = {**spec_kwargs, **kwargs}
    return create_dls_downl(**dl_kwargs)


# --- Simulation benchmarks ---
create_dls_wh = partial(create_dls_from_spec, spec=idb.BenchmarkWH_Simulation)
create_dls_silverbox = partial(create_dls_from_spec, spec=idb.BenchmarkSilverbox_Simulation)
create_dls_cascaded_tanks = partial(create_dls_from_spec, spec=idb.BenchmarkCascadedTanks_Simulation)
create_dls_emps = partial(create_dls_from_spec, spec=idb.BenchmarkEMPS_Simulation)
create_dls_ced = partial(create_dls_from_spec, spec=idb.BenchmarkCED_Simulation)
create_dls_noisy_wh = partial(create_dls_from_spec, spec=idb.BenchmarkNoisyWH_Simulation)
create_dls_robot_forward = partial(create_dls_from_spec, spec=idb.BenchmarkRobotForward_Simulation)
create_dls_robot_inverse = partial(create_dls_from_spec, spec=idb.BenchmarkRobotInverse_Simulation)
create_dls_ship = partial(create_dls_from_spec, spec=idb.BenchmarkShip_Simulation)
create_dls_quad_pelican = partial(create_dls_from_spec, spec=idb.BenchmarkQuadPelican_Simulation)
create_dls_quad_pi = partial(create_dls_from_spec, spec=idb.BenchmarkQuadPi_Simulation)

# --- Prediction benchmarks (Phase 1: no prediction transforms) ---
create_dls_wh_prediction = partial(create_dls_from_spec, spec=idb.BenchmarkWH_Prediction)
create_dls_silverbox_prediction = partial(create_dls_from_spec, spec=idb.BenchmarkSilverbox_Prediction)
create_dls_cascaded_tanks_prediction = partial(create_dls_from_spec, spec=idb.BenchmarkCascadedTanks_Prediction)
create_dls_emps_prediction = partial(create_dls_from_spec, spec=idb.BenchmarkEMPS_Prediction)
create_dls_ced_prediction = partial(create_dls_from_spec, spec=idb.BenchmarkCED_Prediction)
create_dls_noisy_wh_prediction = partial(create_dls_from_spec, spec=idb.BenchmarkNoisyWH_Prediction)
create_dls_robot_forward_prediction = partial(create_dls_from_spec, spec=idb.BenchmarkRobotForward_Prediction)
create_dls_robot_inverse_prediction = partial(create_dls_from_spec, spec=idb.BenchmarkRobotInverse_Prediction)
create_dls_ship_prediction = partial(create_dls_from_spec, spec=idb.BenchmarkShip_Prediction)
create_dls_quad_pelican_prediction = partial(create_dls_from_spec, spec=idb.BenchmarkQuadPelican_Prediction)
create_dls_quad_pi_prediction = partial(create_dls_from_spec, spec=idb.BenchmarkQuadPi_Prediction)

# --- BROAD dataset ---
broad_u_imu_acc = [f"imu_acc{i}" for i in range(3)]
broad_u_imu_gyr = [f"imu_gyr{i}" for i in range(3)]
broad_u_imu_mag = [f"imu_mag{i}" for i in range(3)]
broad_y_opt_pos = [f"opt_pos{i}" for i in range(4)]
broad_y_opt_quat = [f"opt_quat{i}" for i in range(4)]
broad_u = broad_u_imu_acc + broad_u_imu_gyr

create_dls_broad = partial(
    create_dls_downl,
    download_function=idb.datasets.broad.dl_broad,
    u=broad_u,
    y=broad_y_opt_quat,
    win_sz=100,
    stp_sz=30,
)

# --- Collection lists ---
external_datasets_simulation = [
    create_dls_wh,
    create_dls_silverbox,
    create_dls_robot_forward,
    create_dls_noisy_wh,
    create_dls_ced,
    create_dls_emps,
]

external_datasets_prediction = [
    create_dls_cascaded_tanks_prediction,
    create_dls_emps_prediction,
    create_dls_ced_prediction,
    create_dls_noisy_wh_prediction,
    create_dls_quad_pelican_prediction,
    create_dls_quad_pi_prediction,
    create_dls_robot_forward_prediction,
    create_dls_robot_inverse_prediction,
    create_dls_ship_prediction,
    create_dls_silverbox_prediction,
    create_dls_wh_prediction,
]
