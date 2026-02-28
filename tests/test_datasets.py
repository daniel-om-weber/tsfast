"""Tests for tsfast.tsdata module."""
import pytest
import numpy as np


class TestCreateDls:
    def test_simulation_mode(self, dls_simulation):
        batch = dls_simulation.one_batch()
        assert batch[0].shape[-1] == 1  # u only
        assert batch[1].shape[-1] == 1  # y only

    def test_test_dataloader_appended(self, dls_simulation):
        assert len(dls_simulation.loaders) == 3  # train, valid, test

    def test_empty_path_raises_file_not_found(self, tmp_path):
        from tsfast.tsdata import create_dls
        empty_dir = tmp_path / "empty_dataset"
        empty_dir.mkdir()
        with pytest.raises(FileNotFoundError, match="No HDF5 files found"):
            create_dls(u=["u"], y=["y"], dataset=empty_dir,
                       win_sz=100, num_workers=0)


class TestNormalization:
    def test_norm_stats_on_dls(self, dls_simulation):
        from tsfast.tsdata import NormPair, NormStats
        assert hasattr(dls_simulation, 'norm_stats')
        stats = dls_simulation.norm_stats
        assert isinstance(stats, NormStats)
        assert isinstance(stats.u, NormPair)
        assert isinstance(stats.y, NormPair)
        # Named access
        assert stats.u.mean.shape == (1,)
        assert stats.u.std.shape == (1,)
        assert stats.u.min.shape == (1,)
        assert stats.u.max.shape == (1,)
        assert stats.y.mean.shape == (1,)
        assert stats.y.std.shape == (1,)
        # Tuple destructuring still works
        norm_u, norm_y = stats
        assert norm_u.mean.shape == (1,)

    def test_normpair_add(self, dls_simulation):
        stats = dls_simulation.norm_stats
        combined = stats.u + stats.y
        assert combined.mean.shape == (2,)
        assert combined.std.shape == (2,)
        assert combined.min.shape == (2,)
        assert combined.max.shape == (2,)
        np.testing.assert_array_equal(combined.mean, np.hstack([stats.u.mean, stats.y.mean]))
        np.testing.assert_array_equal(combined.std, np.hstack([stats.u.std, stats.y.std]))

    def test_normpair_backward_compat(self, dls_simulation):
        stats = dls_simulation.norm_stats
        # Indexing
        assert np.array_equal(stats.u[0], stats.u.mean)
        assert np.array_equal(stats.u[1], stats.u.std)
        # Iteration
        items = list(stats.u)
        assert len(items) == 4
        # Destructuring
        mean, std, mn, mx = stats.u
        assert np.array_equal(mean, stats.u.mean)

    def test_dls_id_caching(self, wh_path, tmp_path, monkeypatch):
        from tsfast.tsdata import norm as norm_mod
        from tsfast.tsdata import create_dls, NormStats, NormPair
        monkeypatch.setattr(norm_mod, '_cache_path', lambda dls_id: tmp_path / f'{dls_id}.pkl')
        dls = create_dls(
            u=["u"], y=["y"], dataset=wh_path,
            win_sz=100, stp_sz=100, num_workers=0,
            n_batches_train=2, dls_id="test_cache",
        )
        stats1 = dls.norm_stats  # triggers lazy computation + cache write
        assert (tmp_path / 'test_cache.pkl').exists()
        assert isinstance(stats1, NormStats)
        assert isinstance(stats1.u, NormPair)
        # Second call loads from cache
        dls2 = create_dls(
            u=["u"], y=["y"], dataset=wh_path,
            win_sz=100, stp_sz=100, num_workers=0,
            n_batches_train=2, dls_id="test_cache",
        )
        np.testing.assert_array_equal(dls2.norm_stats.u.mean, stats1.u.mean)
        np.testing.assert_array_equal(dls2.norm_stats.u.std, stats1.u.std)
        np.testing.assert_array_equal(dls2.norm_stats.y.mean, stats1.y.mean)

    def test_dls_id_produces_normpair_stats(self, wh_path, tmp_path, monkeypatch):
        from tsfast.tsdata import norm as norm_mod
        from tsfast.tsdata import create_dls, NormPair
        monkeypatch.setattr(norm_mod, '_cache_path', lambda dls_id: tmp_path / f'{dls_id}.pkl')
        dls = create_dls(
            u=["u"], y=["y"], dataset=wh_path,
            win_sz=100, stp_sz=100, num_workers=0,
            n_batches_train=2, dls_id="test_full",
        )
        assert isinstance(dls.norm_stats.u, NormPair)
        assert isinstance(dls.norm_stats.y, NormPair)
        assert dls.norm_stats.u.mean.shape == (1,)
        assert dls.norm_stats.u.min.shape == (1,)
        assert dls.norm_stats.y.mean.shape == (1,)

    def test_is_dataset_directory(self, wh_path):
        from tsfast.tsdata import is_dataset_directory
        assert is_dataset_directory(wh_path) is True
        assert is_dataset_directory(wh_path.parent) is False


class TestSplitByParent:
    def test_split_by_parent_from_path(self, wh_path):
        from tsfast.tsdata import discover_split_files
        splits = discover_split_files(wh_path)
        assert 'train' in splits and 'valid' in splits and 'test' in splits
        assert len(splits['train']) >= 1
        assert len(splits['valid']) >= 1
        assert len(splits['test']) >= 1
        # All files land in exactly one split
        from pathlib import Path
        for f in splits['train']:
            assert Path(f).parent.name == 'train'
        for f in splits['valid']:
            assert Path(f).parent.name == 'valid'
        for f in splits['test']:
            assert Path(f).parent.name == 'test'

    def test_split_by_parent_from_filelist(self, hdf_files):
        from tsfast.tsdata import split_by_parent
        train_idxs, valid_idxs = split_by_parent(hdf_files)
        # All files with train/valid parents should be accounted for
        from pathlib import Path
        expected_train = sum(1 for f in hdf_files if Path(f).parent.name == 'train')
        expected_valid = sum(1 for f in hdf_files if Path(f).parent.name == 'valid')
        assert len(train_idxs) == expected_train
        assert len(valid_idxs) == expected_valid


class TestCreateDlsDict:
    def test_dict_dataset_basic(self, wh_path):
        from tsfast.tsdata import create_dls, discover_split_files
        splits = discover_split_files(wh_path)
        dls = create_dls(
            u=["u"], y=["y"], dataset=splits,
            win_sz=100, stp_sz=100, num_workers=0,
            n_batches_train=2,
        )
        batch = dls.one_batch()
        assert batch[0].shape[-1] == 1
        assert batch[1].shape[-1] == 1

    def test_dict_dataset_has_norm_stats(self, wh_path):
        from tsfast.tsdata import create_dls, discover_split_files, NormStats
        dls = create_dls(
            u=["u"], y=["y"], dataset=discover_split_files(wh_path),
            win_sz=100, stp_sz=100, num_workers=0,
            n_batches_train=2,
        )
        assert hasattr(dls, 'norm_stats')
        assert isinstance(dls.norm_stats, NormStats)

    def test_dict_dataset_test_dl(self, wh_path):
        from tsfast.tsdata import create_dls, discover_split_files
        dls = create_dls(
            u=["u"], y=["y"], dataset=discover_split_files(wh_path),
            win_sz=100, stp_sz=100, num_workers=0,
            n_batches_train=2,
        )
        assert len(dls.loaders) == 3  # train, valid, test

    def test_dict_dataset_no_test(self, wh_path):
        from tsfast.tsdata import create_dls, discover_split_files
        splits = discover_split_files(wh_path)
        del splits['test']
        dls = create_dls(
            u=["u"], y=["y"], dataset=splits,
            win_sz=100, stp_sz=100, num_workers=0,
            n_batches_train=2,
        )
        assert len(dls.loaders) == 2  # train, valid only

    def test_dict_dataset_missing_train_raises(self):
        from tsfast.tsdata import create_dls
        with pytest.raises(ValueError, match="'train'"):
            create_dls(u=["u"], y=["y"], dataset={'valid': ['a.h5']},
                       win_sz=100, num_workers=0)

    def test_dict_dataset_missing_valid_raises(self):
        from tsfast.tsdata import create_dls
        with pytest.raises(ValueError, match="'valid'"):
            create_dls(u=["u"], y=["y"], dataset={'train': ['a.h5']},
                       win_sz=100, num_workers=0)

    def test_dict_equivalent_to_path(self, wh_path):
        """Dict from discover_split_files should produce equivalent results to path-based create_dls."""
        from tsfast.tsdata import create_dls, discover_split_files
        dls_path = create_dls(
            u=["u"], y=["y"], dataset=wh_path,
            win_sz=100, stp_sz=100, num_workers=0,
            n_batches_train=2,
        )
        dls_dict = create_dls(
            u=["u"], y=["y"], dataset=discover_split_files(wh_path),
            win_sz=100, stp_sz=100, num_workers=0,
            n_batches_train=2,
        )
        # Same number of loaders
        assert len(dls_path.loaders) == len(dls_dict.loaders)
        # Same norm stats shapes
        assert dls_path.norm_stats.u.mean.shape == dls_dict.norm_stats.u.mean.shape
        assert dls_path.norm_stats.y.mean.shape == dls_dict.norm_stats.y.mean.shape
