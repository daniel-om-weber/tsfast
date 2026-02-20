"""Shared test fixtures for tsfast test suite."""
import pytest
from pathlib import Path


def _find_project_root():
    d = Path(__file__).resolve().parent.parent
    assert (d / "test_data").is_dir(), f"test_data/ not found at {d}"
    return d


PROJECT_ROOT = _find_project_root()


@pytest.fixture(scope="session")
def wh_path():
    """Path to WienerHammerstein test dataset."""
    p = PROJECT_ROOT / "test_data" / "WienerHammerstein"
    assert p.is_dir(), f"WienerHammerstein test data not found at {p}"
    return p


@pytest.fixture(scope="session")
def pinn_var_ic_path():
    """Path to PINN variable-IC test dataset."""
    p = PROJECT_ROOT / "test_data" / "pinn_var_ic"
    assert p.is_dir(), f"PINN variable-IC test data not found at {p}"
    return p


@pytest.fixture(scope="session")
def hdf_files(wh_path):
    """List of HDF5 files from WienerHammerstein dataset."""
    from tsfast.data.core import get_hdf_files
    return get_hdf_files(wh_path)


@pytest.fixture(scope="session")
def dls_simulation(wh_path):
    """DataLoaders for simulation mode (input-only normalization)."""
    from tsfast.datasets.core import create_dls
    return create_dls(
        u=["u"], y=["y"], dataset=wh_path,
        win_sz=100, stp_sz=100, num_workers=0,
        n_batches_train=10,
    )


@pytest.fixture(scope="session")
def dls_prediction(wh_path):
    """DataLoaders for prediction mode (input+output concatenated)."""
    from tsfast.datasets.core import create_dls
    return create_dls(
        u=["u"], y=["y"], dataset=wh_path,
        win_sz=100, stp_sz=100, num_workers=0,
        n_batches_train=10, prediction=True,
    )


@pytest.fixture(scope="session")
def pinn_path():
    """Path to the PINN mass-spring-damper test dataset."""
    p = PROJECT_ROOT / "test_data" / "pinn"
    assert p.is_dir(), f"PINN test data not found at {p}"
    return p


@pytest.fixture(scope="session")
def orientation_path():
    """Path to the orientation/quaternion test dataset."""
    p = PROJECT_ROOT / "test_data" / "orientation"
    assert p.is_dir(), f"Orientation test data not found at {p}"
    return p


@pytest.fixture(scope="session")
def dls_pinn(pinn_path):
    """DataLoaders for PINN dataset in simulation mode (u -> x,v)."""
    from tsfast.datasets.core import create_dls
    return create_dls(
        u=["u"], y=["x", "v"], dataset=pinn_path,
        win_sz=100, stp_sz=100, num_workers=0,
        n_batches_train=5,
    )


@pytest.fixture(scope="session")
def dls_pinn_prediction(pinn_path):
    """DataLoaders for PINN dataset in prediction mode."""
    from tsfast.datasets.core import create_dls
    return create_dls(
        u=["u"], y=["x", "v"], dataset=pinn_path,
        win_sz=100, stp_sz=100, num_workers=0,
        n_batches_train=5, prediction=True,
    )
