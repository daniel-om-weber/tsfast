"""Pre-configured DataLoader factories for identibench benchmark datasets."""

__all__ = [
    "BENCHMARK_DL_KWARGS",
    "create_dls_wh",
    "create_dls_wh_prediction",
    "create_dls_silverbox",
    "create_dls_silverbox_prediction",
    "create_dls_cascaded_tanks",
    "create_dls_cascaded_tanks_prediction",
    "create_dls_emps",
    "create_dls_emps_prediction",
    "create_dls_ced",
    "create_dls_ced_prediction",
    "create_dls_noisy_wh",
    "create_dls_noisy_wh_prediction",
    "create_dls_robot_forward",
    "create_dls_robot_forward_prediction",
    "create_dls_robot_inverse",
    "create_dls_robot_inverse_prediction",
    "create_dls_ship",
    "create_dls_ship_prediction",
    "create_dls_quad_pelican",
    "create_dls_quad_pelican_prediction",
    "create_dls_quad_pi",
    "create_dls_quad_pi_prediction",
    "broad_u_imu_acc",
    "broad_u_imu_gyr",
    "broad_u_imu_mag",
    "broad_y_opt_pos",
    "broad_y_opt_quat",
    "broad_u",
    "create_dls_broad",
    "create_dls_broad_prediction",
    "external_datasets_simulation",
    "external_datasets_prediction",
    "create_dls_from_spec",
]

from fastai.data.all import *

from ..data import *
from .core import create_dls_downl
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
    spec: idb.benchmark.BenchmarkSpecBase,
    **kwargs,
) -> DataLoaders:
    """Create DataLoaders from an identibench benchmark specification.

    Extracts benchmark-specific kwargs from BENCHMARK_DL_KWARGS and merges them
    with the provided kwargs for create_dls_downl.

    Args:
        spec: benchmark specification from identibench
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


create_dls_wh = partial(create_dls_from_spec, spec=idb.BenchmarkWH_Simulation)
create_dls_wh_prediction = partial(create_dls_from_spec, spec=idb.BenchmarkWH_Prediction)

create_dls_silverbox = partial(create_dls_from_spec, spec=idb.BenchmarkSilverbox_Simulation)
create_dls_silverbox_prediction = partial(create_dls_from_spec, spec=idb.BenchmarkSilverbox_Prediction)

create_dls_cascaded_tanks = partial(create_dls_from_spec, spec=idb.BenchmarkCascadedTanks_Simulation)
create_dls_cascaded_tanks_prediction = partial(create_dls_from_spec, spec=idb.BenchmarkCascadedTanks_Prediction)

create_dls_emps = partial(create_dls_from_spec, spec=idb.BenchmarkEMPS_Simulation)
create_dls_emps_prediction = partial(create_dls_from_spec, spec=idb.BenchmarkEMPS_Prediction)

create_dls_ced = partial(create_dls_from_spec, spec=idb.BenchmarkCED_Simulation)
create_dls_ced_prediction = partial(create_dls_from_spec, spec=idb.BenchmarkCED_Prediction)

create_dls_noisy_wh = partial(create_dls_from_spec, spec=idb.BenchmarkNoisyWH_Simulation)
create_dls_noisy_wh_prediction = partial(create_dls_from_spec, spec=idb.BenchmarkNoisyWH_Prediction)

create_dls_robot_forward = partial(create_dls_from_spec, spec=idb.BenchmarkRobotForward_Simulation)
create_dls_robot_forward_prediction = partial(create_dls_from_spec, spec=idb.BenchmarkRobotForward_Prediction)

create_dls_robot_inverse = partial(create_dls_from_spec, spec=idb.BenchmarkRobotInverse_Simulation)
create_dls_robot_inverse_prediction = partial(create_dls_from_spec, spec=idb.BenchmarkRobotInverse_Prediction)

create_dls_ship = partial(create_dls_from_spec, spec=idb.BenchmarkShip_Simulation)
create_dls_ship_prediction = partial(create_dls_from_spec, spec=idb.BenchmarkShip_Prediction)

create_dls_quad_pelican = partial(create_dls_from_spec, spec=idb.BenchmarkQuadPelican_Simulation)
create_dls_quad_pelican_prediction = partial(create_dls_from_spec, spec=idb.BenchmarkQuadPelican_Prediction)

create_dls_quad_pi = partial(create_dls_from_spec, spec=idb.BenchmarkQuadPi_Simulation)
create_dls_quad_pi_prediction = partial(create_dls_from_spec, spec=idb.BenchmarkQuadPi_Prediction)

broad_u_imu_acc = [f"imu_acc{i}" for i in range(3)]
broad_u_imu_gyr = [f"imu_gyr{i}" for i in range(3)]
broad_u_imu_mag = [f"imu_mag{i}" for i in range(3)]
broad_y_opt_pos = [f"opt_pos{i}" for i in range(4)]
broad_y_opt_quat = [f"opt_quat{i}" for i in range(4)]

broad_u = broad_u_imu_acc + broad_u_imu_gyr

# Todo: add spec, when available
create_dls_broad = partial(
    create_dls_downl,
    download_function=idb.datasets.broad.dl_broad,
    u=broad_u,
    y=broad_y_opt_quat,
    win_sz=100,
    stp_sz=30,
)

create_dls_broad_prediction = partial(
    create_dls_downl,
    download_function=idb.datasets.broad.dl_broad,
    u=broad_u,
    y=broad_y_opt_quat,
    win_sz=100,
    stp_sz=30,
    prediction=True,
)

external_datasets_simulation = [
    create_dls_wh,
    create_dls_silverbox,
    create_dls_robot_forward,
    create_dls_noisy_wh,
    create_dls_ced,
    # create_dls_broad,
    create_dls_emps,
]

external_datasets_prediction = [
    # create_dls_broad_prediction,
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
