"""File discovery and train/valid/test splitting utilities."""

import os
from pathlib import Path

HDF_EXTENSIONS = {".hdf5", ".h5"}


def get_hdf_files(path: Path | str, recurse: bool = True) -> list[Path]:
    """Recursively find .hdf5/.h5 files under path."""
    path = Path(path)
    if not path.exists():
        return []
    if recurse:
        return sorted(f for f in path.rglob("*") if f.suffix in HDF_EXTENSIONS)
    return sorted(f for f in path.iterdir() if f.suffix in HDF_EXTENSIONS)


def discover_split_files(
    path: Path | str,
    train_name: str = "train",
    valid_name: str = "valid",
    test_name: str = "test",
) -> dict[str, list[Path]]:
    """Auto-discover train/valid/test HDF5 files by parent directory name.

    Args:
        path: root directory containing train/valid/test subdirectories
        train_name: name of training subdirectory
        valid_name: name of validation subdirectory
        test_name: name of test subdirectory
    """
    path = Path(path)
    files = get_hdf_files(path)
    return {
        "train": [f for f in files if f.parent.name == train_name],
        "valid": [f for f in files if f.parent.name == valid_name],
        "test": [f for f in files if f.parent.name == test_name],
    }


def split_by_parent(
    files: list,
    train_name: str = "train",
    valid_name: str = "valid",
) -> tuple[list[int], list[int]]:
    """Return (train_indices, valid_indices) based on parent directory names.

    Args:
        files: list of file paths
        train_name: parent directory name for training files
        valid_name: parent directory name for validation files
    """
    train_idxs = [i for i, f in enumerate(files) if Path(f).parent.name == train_name]
    valid_idxs = [i for i, f in enumerate(files) if Path(f).parent.name == valid_name]
    return train_idxs, valid_idxs


def split_by_percentage(files: list, pct: float = 0.8) -> tuple[list[int], list[int]]:
    """Sequential percentage split.

    Args:
        files: list of items to split
        pct: fraction of items assigned to the first split
    """
    split_idx = int(len(files) * pct)
    return list(range(split_idx)), list(range(split_idx, len(files)))


def is_dataset_directory(path: Path | str) -> bool:
    """Check if path contains train/valid/test subdirectories with HDF5 files."""
    for dir_name in ("train", "valid", "test"):
        dir_path = os.path.join(path, dir_name)
        if not os.path.isdir(dir_path):
            return False
        if not get_hdf_files(dir_path):
            return False
    return True
