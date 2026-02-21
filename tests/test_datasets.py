"""Tests for tsfast.datasets module."""
import math
import pytest
import numpy as np
import torch


class TestCreateDls:
    def test_simulation_mode(self, dls_simulation):
        batch = dls_simulation.one_batch()
        assert batch[0].shape[-1] == 1  # u only
        assert batch[1].shape[-1] == 1  # y only

    def test_prediction_mode(self, dls_prediction):
        batch = dls_prediction.one_batch()
        assert batch[0].shape[-1] == 2  # u + y concatenated
        assert batch[1].shape[-1] == 1  # y only

    def test_test_dataloader_appended(self, dls_simulation):
        assert len(dls_simulation.loaders) == 3  # train, valid, test


class TestNormalization:
    def test_norm_stats_on_dls(self, dls_simulation):
        from tsfast.datasets.core import NormPair, NormStats
        assert hasattr(dls_simulation, 'norm_stats')
        stats = dls_simulation.norm_stats
        assert isinstance(stats, NormStats)
        assert isinstance(stats.u, NormPair)
        assert stats.x is None
        assert isinstance(stats.y, NormPair)
        # Named access
        assert stats.u.mean.shape == (1,)
        assert stats.u.std.shape == (1,)
        assert stats.u.min.shape == (1,)
        assert stats.u.max.shape == (1,)
        assert stats.y.mean.shape == (1,)
        assert stats.y.std.shape == (1,)
        # Tuple destructuring still works
        norm_u, norm_x, norm_y = stats
        assert norm_u.mean.shape == (1,)

    def test_extract_mean_std_from_dls(self, dls_simulation):
        from tsfast.datasets.core import extract_mean_std_from_dls
        norm_u, norm_x, norm_y = extract_mean_std_from_dls(dls_simulation)
        assert norm_u.mean.shape == (1,)
        assert norm_u.std.shape == (1,)

    def test_extract_mean_std_from_hdffiles(self, hdf_files):
        from tsfast.datasets.core import extract_mean_std_from_hdffiles
        means, stds = extract_mean_std_from_hdffiles(hdf_files, ["u", "y"])
        assert means.shape == (2,)
        assert stds.shape == (2,)
        assert all(stds > 0)

    def test_estimate_norm_stats(self, dls_simulation):
        from tsfast.datasets.core import estimate_norm_stats, NormPair
        input_stats, output_stats = estimate_norm_stats(dls_simulation, n_batches=3)
        assert isinstance(input_stats, NormPair)
        assert isinstance(output_stats, NormPair)
        assert input_stats.mean.shape == (1,)
        assert input_stats.std.shape == (1,)
        assert input_stats.min.shape == (1,)
        assert input_stats.max.shape == (1,)
        assert all(input_stats.std > 0)
        assert all(input_stats.min <= input_stats.mean)
        assert all(input_stats.max >= input_stats.mean)

    def test_normpair_add(self, dls_simulation):
        from tsfast.datasets.core import NormPair
        stats = dls_simulation.norm_stats
        combined = stats.u + stats.y
        assert combined.mean.shape == (2,)
        assert combined.std.shape == (2,)
        assert combined.min.shape == (2,)
        assert combined.max.shape == (2,)
        np.testing.assert_array_equal(combined.mean, np.hstack([stats.u.mean, stats.y.mean]))
        np.testing.assert_array_equal(combined.std, np.hstack([stats.u.std, stats.y.std]))

    def test_normpair_backward_compat(self, dls_simulation):
        from tsfast.datasets.core import NormPair
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

    def test_extract_norm_from_hdffiles(self, hdf_files):
        from tsfast.datasets.core import extract_norm_from_hdffiles, NormPair
        from pathlib import Path
        train_files = [f for f in hdf_files if Path(f).parent.name == 'train']
        result = extract_norm_from_hdffiles(train_files, ["u", "y"])
        assert isinstance(result, NormPair)
        assert result.mean.shape == (2,)
        assert result.std.shape == (2,)
        assert result.min.shape == (2,)
        assert result.max.shape == (2,)
        assert all(result.std > 0)
        assert all(result.min <= result.mean)
        assert all(result.max >= result.mean)

    def test_extract_norm_from_hdffiles_empty(self):
        from tsfast.datasets.core import extract_norm_from_hdffiles
        assert extract_norm_from_hdffiles([], []) is None

    def test_dls_id_caching(self, wh_path, tmp_path, monkeypatch):
        from tsfast.datasets import core as ds_core
        from tsfast.datasets.core import create_dls, NormStats, NormPair
        monkeypatch.setattr(ds_core, '_cache_path', lambda dls_id: tmp_path / f'{dls_id}.pkl')
        dls = create_dls(
            u=["u"], y=["y"], dataset=wh_path,
            win_sz=100, stp_sz=100, num_workers=0,
            n_batches_train=2, dls_id="test_cache",
        )
        assert (tmp_path / 'test_cache.pkl').exists()
        stats1 = dls.norm_stats
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
        from tsfast.datasets import core as ds_core
        from tsfast.datasets.core import create_dls, NormPair
        monkeypatch.setattr(ds_core, '_cache_path', lambda dls_id: tmp_path / f'{dls_id}.pkl')
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
        from tsfast.datasets.core import is_dataset_directory
        assert is_dataset_directory(wh_path) is True
        assert is_dataset_directory(wh_path.parent) is False


class TestSplitByParent:
    def test_split_by_parent_from_path(self, wh_path):
        from tsfast.datasets.core import split_by_parent
        splits = split_by_parent(wh_path)
        assert 'train' in splits and 'valid' in splits and 'test' in splits
        assert len(splits['train']) >= 1
        assert len(splits['valid']) >= 1
        assert len(splits['test']) >= 1
        # All files land in exactly one split
        from pathlib import Path
        for f in splits['train']: assert Path(f).parent.name == 'train'
        for f in splits['valid']: assert Path(f).parent.name == 'valid'
        for f in splits['test']:  assert Path(f).parent.name == 'test'

    def test_split_by_parent_from_filelist(self, hdf_files):
        from tsfast.datasets.core import split_by_parent
        splits = split_by_parent(hdf_files)
        total = len(splits['train']) + len(splits['valid']) + len(splits['test'])
        assert total == len(hdf_files)


class TestCreateDlsDict:
    def test_dict_dataset_basic(self, wh_path):
        from tsfast.datasets.core import create_dls, split_by_parent
        splits = split_by_parent(wh_path)
        dls = create_dls(
            u=["u"], y=["y"], dataset=splits,
            win_sz=100, stp_sz=100, num_workers=0,
            n_batches_train=2,
        )
        batch = dls.one_batch()
        assert batch[0].shape[-1] == 1
        assert batch[1].shape[-1] == 1

    def test_dict_dataset_has_norm_stats(self, wh_path):
        from tsfast.datasets.core import create_dls, split_by_parent, NormStats
        dls = create_dls(
            u=["u"], y=["y"], dataset=split_by_parent(wh_path),
            win_sz=100, stp_sz=100, num_workers=0,
            n_batches_train=2,
        )
        assert hasattr(dls, 'norm_stats')
        assert isinstance(dls.norm_stats, NormStats)

    def test_dict_dataset_test_dl(self, wh_path):
        from tsfast.datasets.core import create_dls, split_by_parent
        dls = create_dls(
            u=["u"], y=["y"], dataset=split_by_parent(wh_path),
            win_sz=100, stp_sz=100, num_workers=0,
            n_batches_train=2,
        )
        assert len(dls.loaders) == 3  # train, valid, test

    def test_dict_dataset_no_test(self, wh_path):
        from tsfast.datasets.core import create_dls, split_by_parent
        splits = split_by_parent(wh_path)
        del splits['test']
        dls = create_dls(
            u=["u"], y=["y"], dataset=splits,
            win_sz=100, stp_sz=100, num_workers=0,
            n_batches_train=2,
        )
        assert len(dls.loaders) == 2  # train, valid only

    def test_dict_dataset_missing_train_raises(self):
        from tsfast.datasets.core import create_dls
        with pytest.raises(ValueError, match="'train'"):
            create_dls(u=["u"], y=["y"], dataset={'valid': ['a.h5']},
                       win_sz=100, num_workers=0)

    def test_dict_dataset_missing_valid_raises(self):
        from tsfast.datasets.core import create_dls
        with pytest.raises(ValueError, match="'valid'"):
            create_dls(u=["u"], y=["y"], dataset={'train': ['a.h5']},
                       win_sz=100, num_workers=0)

    def test_dict_equivalent_to_path(self, wh_path):
        """Dict from split_by_parent should produce equivalent results to path-based create_dls."""
        from tsfast.datasets.core import create_dls, split_by_parent
        dls_path = create_dls(
            u=["u"], y=["y"], dataset=wh_path,
            win_sz=100, stp_sz=100, num_workers=0,
            n_batches_train=2,
        )
        dls_dict = create_dls(
            u=["u"], y=["y"], dataset=split_by_parent(wh_path),
            win_sz=100, stp_sz=100, num_workers=0,
            n_batches_train=2,
        )
        # Same number of loaders
        assert len(dls_path.loaders) == len(dls_dict.loaders)
        # Same norm stats shapes
        assert dls_path.norm_stats.u.mean.shape == dls_dict.norm_stats.u.mean.shape
        assert dls_path.norm_stats.y.mean.shape == dls_dict.norm_stats.y.mean.shape


class TestTbpttDataLoader:
    def test_tbptt_dl_sub_sequence_shape(self, wh_path):
        from tsfast.datasets.core import create_dls
        dls = create_dls(
            u=["u"], y=["y"], dataset=wh_path,
            win_sz=100, stp_sz=100, num_workers=0,
            n_batches_train=5, sub_seq_len=50,
        )
        batch = dls.one_batch()
        assert batch[0].shape[1] == 50  # sub_seq_len truncation

    def test_tbptt_dl_n_sub_seq(self, wh_path):
        from tsfast.datasets.core import create_dls
        dls = create_dls(
            u=["u"], y=["y"], dataset=wh_path,
            win_sz=100, stp_sz=100, num_workers=0,
            n_batches_train=5, sub_seq_len=50,
        )
        # win_sz=100 / sub_seq_len=50 = 2 sub-sequences per base batch
        assert dls.train.n_sub_seq == 2

    @pytest.mark.slow
    def test_tbptt_rnn_training(self, wh_path):
        from tsfast.datasets.core import create_dls
        from tsfast.models.rnn import RNNLearner
        dls = create_dls(
            u=["u"], y=["y"], dataset=wh_path,
            win_sz=100, stp_sz=100, num_workers=0,
            n_batches_train=2, sub_seq_len=25,
        )
        lrn = RNNLearner(dls, rnn_type="gru", num_layers=1, hidden_size=10, stateful=True)
        lrn.fit(1, 1e-4)
        assert not math.isnan(lrn.recorder.values[-1][1])

    @pytest.mark.slow
    def test_tbptt_multiworker_rnn_reset(self, wh_path):
        from tsfast.datasets.core import create_dls
        dls = create_dls(
            u=["u"], y=["y"], dataset=wh_path,
            win_sz=100, stp_sz=100, num_workers=2,
            n_batches_train=4, sub_seq_len=50,
        )
        dl = dls.train
        n_sub_seq = dl.n_sub_seq
        resets = []
        for batch in dl:
            resets.append(dl.rnn_reset)
        # rnn_reset should be True at sub-sequence boundaries (every n_sub_seq batches)
        for i, r in enumerate(resets):
            expected = (i % n_sub_seq) == 0
            assert r == expected, f"batch {i}: rnn_reset={r}, expected={expected}"

    def test_batch_limit_factory(self, wh_path):
        from tsfast.datasets.core import create_dls
        with pytest.warns(DeprecationWarning, match="max_batches_training"):
            dls = create_dls(
                u=["u"], y=["y"], dataset=wh_path,
                win_sz=100, stp_sz=100, num_workers=0,
                n_batches_train=None, max_batches_training=3,
            )
        assert len(dls.train) <= 3


class TestWeightedSampling:
    def test_uniform_p_of_category(self):
        import pandas as pd
        from tsfast.data.loader import uniform_p_of_category
        df = pd.DataFrame({
            'category': ['A'] * 10 + ['B'] * 30 + ['C'] * 60,
            'value': range(100),
        })
        result = df.pipe(uniform_p_of_category('category'))
        assert result.p_sample.sum() == pytest.approx(1.0)
        # Each category should have equal total weight (1/3)
        for cat in ['A', 'B', 'C']:
            cat_weight = result[result.category == cat].p_sample.sum()
            assert cat_weight == pytest.approx(1 / 3, abs=1e-10)

    def test_uniform_p_of_float(self):
        import pandas as pd
        from tsfast.data.loader import uniform_p_of_float
        df = pd.DataFrame({'speed': np.linspace(0, 100, 200)})
        result = df.pipe(uniform_p_of_float('speed', bins=5))
        assert result.p_sample.sum() == pytest.approx(1.0)

    def test_uniform_p_of_float_with_gaps(self):
        import pandas as pd
        from tsfast.data.loader import uniform_p_of_float_with_gaps
        # Create data with gaps: cluster around 0-10 and 90-100
        values = np.concatenate([np.random.uniform(0, 10, 80), np.random.uniform(90, 100, 20)])
        df = pd.DataFrame({'altitude': values})
        result = df.pipe(uniform_p_of_float_with_gaps('altitude', bins=10))
        assert result.p_sample.sum() == pytest.approx(1.0)

    def test_p_of_category_chaining(self):
        import pandas as pd
        from tsfast.data.loader import uniform_p_of_category
        df = pd.DataFrame({
            'color': ['red'] * 20 + ['blue'] * 80,
            'size': ['S'] * 50 + ['L'] * 50,
        })
        result = df.pipe(uniform_p_of_category('color')).pipe(uniform_p_of_category('size'))
        assert result.p_sample.sum() == pytest.approx(1.0)
