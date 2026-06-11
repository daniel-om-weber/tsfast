"""Pre-configured DataLoader factories for identibench benchmark datasets."""

__all__ = [
    "create_dls_from_spec",
    "create_dls_wh",
    "create_dls_silverbox",
    "create_dls_cascaded_tanks",
    "create_dls_emps",
    "create_dls_ced",
    "create_dls_noisy_wh",
    "create_dls_robot_forward",
    "create_dls_robot_inverse",
    "create_dls_ship",
    "create_dls_quad_pelican",
    "create_dls_quad_pi",
    "create_dls_wh_prediction",
    "create_dls_silverbox_prediction",
    "create_dls_cascaded_tanks_prediction",
    "create_dls_emps_prediction",
    "create_dls_ced_prediction",
    "create_dls_noisy_wh_prediction",
    "create_dls_robot_forward_prediction",
    "create_dls_robot_inverse_prediction",
    "create_dls_ship_prediction",
    "create_dls_quad_pelican_prediction",
    "create_dls_quad_pi_prediction",
    "create_dls_riann",
    "external_datasets_simulation",
    "external_datasets_prediction",
]

from functools import partial

import identibench as idb

from .pipeline import DataLoaders, create_dls

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
    # Orientation Estimation
    "BenchmarkRIANN_Inclination": {"win_sz": 100, "stp_sz": 30},
}


def create_dls_from_spec(
    spec: idb.BenchmarkSpec,
    **kwargs,
) -> DataLoaders:
    """Create DataLoaders from an identibench benchmark specification.

    Download/preparation is delegated to identibench (`spec.ensure_datasets_exist`,
    cached under the identibench data root). Files are resolved through the spec's
    role accessors (`spec.train_files`/`valid_files`/`test_files`), never by
    parsing the on-disk layout; the test split is the union of the spec's named
    test sets. Evaluation parameters are read off `spec.task` (e.g. window sizing
    for prediction benchmarks).

    Requires a trainable spec (the workshop/robot/quad/ship benchmarks and the
    combined RIANN orientation benchmark). All-test evaluation specs
    (per-source orientation, DFJIMU) have no DataLoader factory here.

    Args:
        spec: benchmark specification from identibench

    Raises:
        ValueError: If the spec defines no train/valid split (`spec.is_trainable`
            is False).
    """
    if not spec.is_trainable:
        raise ValueError(
            f"{spec.name} is an all-test evaluation spec (no train/valid split); "
            "it cannot provide training DataLoaders."
        )
    spec.ensure_datasets_exist()

    dataset = {
        "train": [str(f) for f in spec.train_files()],
        "valid": [str(f) for f in spec.valid_files()],
        "test": [str(f) for f in spec.test_files()],
    }
    spec_kwargs = {
        "u": spec.u_cols,
        "y": spec.y_cols,
        "dataset": dataset,
    }

    # Prediction specs: in Phase 1 we only handle the window sizing, not the
    # actual prediction input concatenation (deferred to Phase 3).
    if isinstance(spec.task, idb.Prediction):
        spec_kwargs.update(
            {
                "win_sz": spec.task.horizon + spec.task.init_window,
                "valid_stp_sz": spec.task.step,
            }
        )

    if spec.name in BENCHMARK_DL_KWARGS:
        spec_kwargs.update(BENCHMARK_DL_KWARGS[spec.name])

    dl_kwargs = {**spec_kwargs, **kwargs}
    return create_dls(**dl_kwargs)


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

# --- Orientation estimation (IMU inclination) ---
# Only the combined RIANN corpus carries train/valid roles (the paper's
# cross-dataset split); the per-source specs (BROAD, TUM-VI, OxIOD, EuRoC,
# RepoIMU, Caruso, DFJIMU) are all-test evaluation benchmarks.
create_dls_riann = partial(create_dls_from_spec, spec=idb.BenchmarkRIANN_Inclination)


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
