"""HDF5 data extraction, resampling, and tensor types for time series."""

__all__ = [
    "hdf_extensions",
    "obj_in_lst",
    "count_parameters",
    "get_hdf_files",
    "apply_df_tfms",
    "CreateDict",
    "ValidClmContains",
    "ValidClmIs",
    "FilterClm",
    "get_hdf_seq_len",
    "df_get_hdf_seq_len",
    "DfHDFGetSeqLen",
    "DfResamplingFactor",
    "DfHDFCreateWindows",
    "DfApplyFuncSplit",
    "DfFilterQuery",
    "DfDropClmExcept",
    "calc_shift_offsets",
    "running_mean",
    "downsample_mean",
    "resample_interp",
    "hdf_extract_sequence",
    "Memoize",
    "MemoizeMP",
    "HDF2Sequence",
    "hdf_attrs2scalars",
    "HDF_Attrs2Scalars",
    "hdf_ds2scalars",
    "HDF_DS2Scalars",
    "TensorSequences",
    "TensorSequencesInput",
    "TensorSequencesOutput",
    "toTensorSequencesInput",
    "toTensorSequencesOutput",
    "TensorScalars",
    "TensorScalarsInput",
    "TensorScalarsOutput",
    "toTensorScalarsInput",
    "toTensorScalarsOutput",
    "plot_sequence",
    "plot_seqs_single_figure",
    "plot_seqs_multi_figures",
    "show_batch",
    "show_results",
]

from fastai.data.all import *
import h5py


def obj_in_lst(lst: list, cls: type):
    """Retrieve first object of type cls from a list."""
    return next(o for o in lst if type(o) is cls)


def count_parameters(model: nn.Module) -> int:
    """Retrieve number of trainable parameters of a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


hdf_extensions = [".hdf5", ".h5"]


def get_hdf_files(path: Path, recurse: bool = True, folders: list | None = None) -> L:
    "Get hdf5 files in `path` recursively, only in `folders`, if specified."
    return get_files(path, extensions=hdf_extensions, recurse=recurse, folders=folders)


def apply_df_tfms(src, pd_tfms: list | None = None) -> pd.DataFrame:
    """Create Pandas DataFrame from a list of items, with a list of DataFrame transforms applied."""
    if type(src) is pd.DataFrame:
        df = src
    else:
        df = pd.DataFrame(data=src.items, columns=["path"], dtype=str)
    if pd_tfms is not None:
        for t in pd_tfms:
            df = t(df)
    return df


def CreateDict(pd_tfms: list | None = None):
    """Create list of dictionaries from a list of items, with a list of DataFrame transforms applied."""

    def _inner(src):
        df = apply_df_tfms(src, pd_tfms)
        #         df_dict_list = df.to_dict(orient='records') native to_dict is slower than self written approach
        df_values = df.values
        df_dict = {name: list(df_values[:, i]) for (i, name) in enumerate(df.columns)}
        df_dict_list = [{name: df_dict[name][i] for name in df_dict} for i in range(len(df))]
        return df_dict_list

    return _inner


def ValidClmContains(lst_valid: list[str]):
    """Add validation column using a list of strings that are part of the validation frames."""

    def _inner(df):
        re_valid = "|".join([re.escape(f) for f in lst_valid])
        df["valid"] = df.path.str.contains(re_valid)
        return df

    return _inner


def ValidClmIs(lst_valid: list):
    """Add validation column using a list of validation filenames."""

    def _inner(df):
        df["valid"] = df.path.isin([str(f) for f in lst_valid])
        return df

    return _inner


def FilterClm(clm_name: str, func: Callable = lambda x: x):
    """Filter DataFrame rows by applying a function to a column."""

    def _inner(df):
        return df[func(df[clm_name])]

    return _inner


def get_hdf_seq_len(df, clm: str, ds=None) -> int:
    """Extract the sequence length of the dataset with the given column name and file path."""
    with h5py.File(df["path"], "r") as f:
        ds = f if "dataset" not in df else f[df["dataset"]]
        f_len = max(ds[clm].shape)
    return f_len


def df_get_hdf_seq_len(df: pd.DataFrame, clm: str, ds=None) -> pd.DataFrame:
    """Extract the sequence length of every file to prepare window extractions with DfHDFCreateWindows."""
    #     df['seq_len'] = ([get_hdf_seq_len(row.path,clm) for (idx, row) in df.iterrows()])
    df["seq_len"] = df.apply(lambda x: get_hdf_seq_len(x, clm), axis=1)
    return df


def DfHDFGetSeqLen(clm: str):
    """DataFrame transform that adds sequence length column from HDF5 files."""

    def _inner(df):
        return df_get_hdf_seq_len(df, clm)

    return _inner


import numbers


def DfResamplingFactor(src_fs: float | str, lst_targ_fs: list[float]):
    """DataFrame transform that expands rows with resampling factors for target frequencies."""
    if not isinstance(src_fs, numbers.Number) and type(src_fs) is not str:
        raise ValueError("src_fs has to be a column name or a fixed number")

    def _inner(df):
        np_targ_fs = np.array(lst_targ_fs)
        pd.options.mode.chained_assignment = (
            None  # every row is a reference so we need to suppress the warning messages while copying
        )

        # repeat entries for every target fs
        res_df = df.iloc[np.repeat(np.arange(len(df)), len(np_targ_fs))]
        targ_fs = np.tile(np_targ_fs, len(df))
        res_df["targ_fs"] = targ_fs

        if isinstance(src_fs, numbers.Number):
            # src_fs is a fixed number
            res_df["resampling_factor"] = targ_fs / src_fs
        else:
            # src_fs is a column name of the df
            res_df["resampling_factor"] = targ_fs / res_df[src_fs]

        pd.options.mode.chained_assignment = "warn"

        return res_df

    return _inner


def DfHDFCreateWindows(win_sz: int, stp_sz: int, clm: str, fixed_start: bool = False, fixed_end: bool = False):
    """Create windows of sequences, splitting each sequence into multiple items."""

    def _inner(df):
        if fixed_start and fixed_end:
            raise Exception

        if "seq_len" in df:
            np_f_len = df.seq_len.values
        else:
            np_f_len = np.array([get_hdf_seq_len(row, clm) for (idx, row) in df.iterrows()])

        if "resampling_factor" in df:
            np_f_len = (np_f_len * df.resampling_factor.values).astype(int)

        n_win = ((np_f_len - win_sz) // stp_sz) + 1
        # cast array n_win to int and clip negative values to 0
        n_win = n_win.astype(int)
        n_win = np.clip(
            n_win, a_min=0, a_max=None
        )  # remove negative values at instances where the winsize is smaller than the seq_len
        lst_idx = np.arange(len(np_f_len))

        pd.options.mode.chained_assignment = (
            None  # every row is a reference so we need to suppress the warning messages while copying
        )

        res_df = df.iloc[np.repeat(lst_idx, n_win)]
        #         res_df = df.loc[np.repeat(lst_idx,n_win)] #the loc variant as a little bit slower because it creates copies and returns wrong values with redundant indexes, but is more robust

        step_idx = np.concatenate([np.arange(x) for x in n_win])

        res_df["l_slc"] = step_idx * stp_sz if not fixed_start else None
        res_df["r_slc"] = step_idx * stp_sz + win_sz if not fixed_end else None

        pd.options.mode.chained_assignment = "warn"

        return res_df

    return _inner


def DfApplyFuncSplit(split_func: Callable, func1: Callable, func2: Callable):
    """Apply different DataFrame transforms per split.

    Args:
        split_func: train/validation split function returning two index lists
        func1: transform applied to the first split indices
        func2: transform applied to the second split indices
    """

    def _inner(df):
        (idxs1, idxs2) = split_func(df.path)
        df1 = func1(df.iloc[idxs1])
        df2 = func2(df.iloc[idxs2])
        return pd.concat((df1, df2))

    return _inner


def DfFilterQuery(query: str):
    """DataFrame transform that filters rows using a pandas query string."""

    def _inner(df):
        return df.query(query)

    return _inner


def DfDropClmExcept(clms: list[str] = ["path", "l_slc", "r_slc", "p_sample", "resampling_factor"]):
    """Drop unused DataFrame columns to accelerate dictionary conversion."""

    def _inner(df):
        return df[[c for c in clms if c in df]]

    return _inner


def calc_shift_offsets(clm_shift: list[int]):
    """Compute left/right shift offsets and dimension reduction from column shifts."""
    clm_shift = np.array(clm_shift)
    l_offs = -min(clm_shift.min(), 0)
    r_offs = -max(clm_shift.max(), 0)
    l_shift = clm_shift + l_offs
    r_shift = clm_shift + r_offs
    dim_red = l_offs - r_offs
    return l_shift, r_shift, dim_red


def running_mean(x: np.ndarray, N: int) -> np.ndarray:
    """Compute running mean with window size N."""
    cumsum = np.cumsum(np.insert(x, 0, 0, axis=0), axis=0)
    return (cumsum[N:] - cumsum[:-N]) / float(N)


def downsample_mean(x: np.ndarray, N: int) -> np.ndarray:
    """Downsample by averaging consecutive groups of N samples."""
    trunc = -(x.shape[0] % N)
    trunc = trunc if trunc != 0 else None
    return x[:trunc, :].reshape((-1, N, x.shape[-1])).mean(axis=1)


from scipy.signal import butter, lfilter, lfilter_zi


def resample_interp(
    x: np.ndarray,
    resampling_factor: float,
    sequence_first: bool = True,
    lowpass_cut: float = 1.0,
    upsample_cubic_cut: float | None = None,
) -> np.ndarray:
    """Signal resampling using linear or cubic interpolation.

    Args:
        x: signal to resample with shape features x resampling_dimension
            or resampling_dimension x features if sequence_first=True
        resampling_factor: factor > 0 that scales the signal
        sequence_first: whether the resampling dimension is the first axis
        lowpass_cut: upper boundary for resampling_factor that activates the
            lowpass filter, low values exchange accuracy for performance
        upsample_cubic_cut: lower boundary for resampling_factor that activates
            cubic interpolation at high upsampling values, None deactivates it
    """

    if sequence_first:
        x = x.T

    fs_n = resampling_factor
    # if downsampling rate is too high, lowpass filter before interpolation
    if fs_n < lowpass_cut:
        b, a = butter(2, fs_n)
        zi = lfilter_zi(b, a) * x[:, :1]  # initialize filter with steady state at first time step value
        x, _ = lfilter(b, a, x, axis=-1, zi=zi)

    #         sos = butter(2, fs_n*1.2,output='sos')
    # #         sos = signal.cheby2(2,20, fs_n,output='sos')
    # #         import pdb;pdb.set_trace()
    #         zi = np.swapaxes(signal.sosfilt_zi(sos)[...,None]*x[:,0],1,2)
    #         x,_ = signal.sosfilt(sos, x,axis=-1,zi=zi)

    x_int = tensor(x)[None, ...]
    targ_size = int(x.shape[-1] * fs_n)

    #     if upsampling rate is too high, switch from linear to cubic interpolation
    if upsample_cubic_cut is None or fs_n <= upsample_cubic_cut:
        x = nn.functional.interpolate(x_int, size=targ_size, mode="linear", align_corners=False)[0].numpy()
    else:
        x = nn.functional.interpolate(x_int[..., None], size=[targ_size, 1], mode="bicubic", align_corners=False)[
            0, ..., 0
        ].numpy()
    #     x = array(x_int)[0]

    if sequence_first:
        x = x.T

    return x


from scipy.signal import resample


def hdf_extract_sequence(
    hdf_path: str | Path,
    clms: list[str],
    dataset: str | None = None,
    l_slc: int | None = None,
    r_slc: int | None = None,
    resampling_factor: float | None = None,
    fs_idx: int | None = None,
    dt_idx: int | bool = False,
    fast_resample: bool = True,
) -> np.ndarray:
    """Extract a sequence with shape [seq_len x num_features] from an HDF5 file.

    Args:
        hdf_path: file path of hdf file, may be a string or path type
        clms: list of dataset names of sequences in hdf file
        dataset: dataset root for clms, useful for multiple sequences stored in one file
        l_slc: left boundary for extraction of a window of the whole sequence
        r_slc: right boundary for extraction of a window of the whole sequence
        resampling_factor: scaling factor for the sequence length, uses resample_interp
        fs_idx: clms list idx of fs entry, scaled by resampling_factor after resampling
        dt_idx: clms list idx of dt entry, scaled by resampling_factor after resampling
        fast_resample: if True, uses linear interpolation with anti-aliasing filter instead of fft
    """

    if resampling_factor is not None:
        seq_len = (
            r_slc - l_slc if l_slc is not None and r_slc is not None else None
        )  # calculate seq_len for later slicing, necesary because of rounding errors in resampling
        if l_slc is not None:
            l_slc = math.floor(l_slc / resampling_factor)
        if r_slc is not None:
            r_slc = math.ceil(r_slc / resampling_factor)

    with h5py.File(hdf_path, "r") as f:
        ds = f if dataset is None else f[dataset]
        l_array = [(ds[n][l_slc:r_slc]) for n in clms]
        seq = np.stack(l_array, axis=-1)

    if resampling_factor is not None:
        if fast_resample:
            res_seq = resample_interp(seq, resampling_factor)
        else:
            res_seq = resample(seq, int(seq.shape[0] * resampling_factor), window=("kaiser", 14.0))

        if fs_idx is not None:
            res_seq[:, fs_idx] = seq[0, fs_idx] * resampling_factor
        if dt_idx is not None:
            res_seq[:, dt_idx] = seq[0, dt_idx] / resampling_factor

        seq = res_seq

        if seq_len is not None:
            seq = seq[:seq_len]  # cut the part of the sequence that is too long because of resampling rounding errors

    return seq


class Memoize:
    """Single-process memoization cache for function results.

    Args:
        fn: function to memoize
    """

    def __init__(self, fn: Callable):
        self.fn = fn
        self.memo = {}

    def __call__(self, *args):
        if args not in self.memo:
            self.memo[args] = self.fn(*args)
        return self.memo[args]


from multiprocessing import Lock, Manager, shared_memory


class MemoizeMP:
    """Multi-process memoization cache using shared memory.

    Args:
        fn: function to memoize
    """

    def __init__(self, fn: Callable):
        self.fn = fn
        self.manager = Manager()
        self.results_dict = self.manager.dict()  # Stores metadata about computed results
        self.lock = Lock()  # Ensure atomic updates to the results_dict
        self.local_memo = {}  # Local cache for each process
        self.shared_memory_segments = []  # Track all shared memory segments

    def __call__(self, *args):
        if args in self.local_memo:
            return self.local_memo[args][0]
        with self.lock:
            if args in self.results_dict:
                result_info = self.results_dict[args]
                existing_shm = shared_memory.SharedMemory(name=result_info["name"])
                result = np.ndarray(result_info["shape"], dtype=result_info["dtype"], buffer=existing_shm.buf)
                self.local_memo[args] = (result, existing_shm)
                return result
        result = self.fn(*args)
        with self.lock:
            if args not in self.results_dict:
                result_shm = shared_memory.SharedMemory(create=True, size=result.nbytes)
                shm_array = np.ndarray(result.shape, dtype=result.dtype, buffer=result_shm.buf)
                shm_array[:] = result[:]

                self.results_dict[args] = {"name": result_shm.name, "shape": result.shape, "dtype": result.dtype.str}
                self.local_memo[args] = (result, result_shm)
                # Track this shared memory segment for later cleanup
                self.shared_memory_segments.append(result_shm)
        return result

    def cleanup_shared_memory(self):
        """Explicitly cleanup all tracked shared memory segments."""
        for shm in self.shared_memory_segments:
            try:
                shm.close()
                shm.unlink()
            except FileNotFoundError:
                # The shared memory segment was already cleaned up
                pass
        # Clear the list after cleanup
        self.shared_memory_segments.clear()

    def __del__(self):
        self.cleanup_shared_memory()


class HDF2Sequence(Transform):
    """Transform that extracts sequences from HDF5 files.

    Args:
        clm_names: list of dataset column names to extract
        clm_shift: per-column integer shifts for temporal alignment
        truncate_sz: number of leading samples to truncate after shifting
        to_cls: callable to convert the resulting array to a specific type
        cached: True for memoized caching, False to disable, or "local" for local-only cache
        fs_idx: index of frequency column, scaled by resampling_factor after resampling
        dt_idx: index of time column, scaled by resampling_factor after resampling
        fast_resample: use linear interpolation vs fft resampling
    """

    def __init__(
        self,
        clm_names: list[str],
        clm_shift: list[int] | None = None,
        truncate_sz: int | None = None,
        to_cls: Callable = noop,
        cached: bool | str = True,
        fs_idx: int | None = None,
        dt_idx: int | None = None,
        fast_resample: bool = True,
    ):
        if clm_shift is not None:
            assert len(clm_shift) == len(clm_names) and all(isinstance(n, int) for n in clm_shift)
            self.l_shift, self.r_shift, _ = calc_shift_offsets(clm_shift)

        if not cached:
            self._exseq = self._hdf_extract_sequence
        elif cached == "local":
            self._exseq = Memoize(self._hdf_extract_sequence)
        else:
            self._exseq = Memoize(self._hdf_extract_sequence)
            # self._exseq = MemoizeMP(self._hdf_extract_sequence)

        self.cached = cached is not None
        self.clm_names = clm_names
        self.clm_shift = clm_shift
        self.truncate_sz = truncate_sz
        self.to_cls = to_cls
        self.fs_idx = fs_idx
        self.dt_idx = dt_idx
        self.fast_resample = fast_resample

    def _hdf_extract_sequence(
        self,
        hdf_path,
        dataset=None,
        l_slc=None,
        r_slc=None,
        resampling_factor=None,
        fs_idx=None,
        dt_idx=None,
        fast_resample=True,
    ):
        """Extract a sequence with shape [seq_len x num_features]."""

        if resampling_factor is not None:
            seq_len = (
                r_slc - l_slc if l_slc is not None and r_slc is not None else None
            )  # calculate seq_len for later slicing, necesary because of rounding errors in resampling
            if l_slc is not None:
                l_slc = math.floor(l_slc / resampling_factor)
            if r_slc is not None:
                r_slc = math.ceil(r_slc / resampling_factor)

        with h5py.File(hdf_path, "r") as f:
            ds = f if dataset is None else f[dataset]
            l_array = [(ds[n][l_slc:r_slc]) for n in self.clm_names]
            seq = np.stack(l_array, axis=-1)

        if resampling_factor is not None:
            if fast_resample:
                res_seq = resample_interp(seq, resampling_factor)
            else:
                res_seq = resample(seq, int(seq.shape[0] * resampling_factor), window=("kaiser", 14.0))

            if fs_idx is not None:
                res_seq[:, fs_idx] = seq[0, fs_idx] * resampling_factor
            if dt_idx is not None:
                res_seq[:, dt_idx] = seq[0, dt_idx] / resampling_factor
            seq = res_seq

            if seq_len is not None:
                seq = seq[
                    :seq_len
                ]  # cut the part of the sequence that is too long because of resampling rounding errors

        return seq

    def _extract_dict_sequence(self, item):
        if hasattr(item, "keys"):
            path = item["path"]
            dataset = item["dataset"] if "dataset" in item else None
            l_slc = item["l_slc"] if "l_slc" in item else None
            r_slc = item["r_slc"] if "r_slc" in item else None
            resampling_factor = item["resampling_factor"] if "resampling_factor" in item else None

            if self.cached:
                seq = self._exseq(
                    path, dataset, None, None, resampling_factor, self.fs_idx, self.dt_idx, self.fast_resample
                )[l_slc:r_slc]
            else:
                seq = self._exseq(
                    path, dataset, l_slc, r_slc, resampling_factor, self.fs_idx, self.dt_idx, self.fast_resample
                )
        else:
            seq = self._exseq(str(item), None, None, None, None, None)

        # shift clms of result by given value
        if self.clm_shift is not None:
            l_seq = seq.shape[0]
            seq = np.stack([seq[self.l_shift[i] : l_seq + self.r_shift[i], i] for i in range(seq.shape[1])], axis=-1)

        if self.truncate_sz is not None:
            seq = seq[self.truncate_sz :]

        # it is important to slice first and then do the class conversion
        #         return self.to_cls(seq.astype('f8'))#workaround for random bug, that mitigates convergence if the numpy array is an f4 array. Seems to make no sense because the result does not change.
        return self.to_cls(seq)

    def encodes(self, item) -> None:
        return self._extract_dict_sequence(item)


def hdf_attrs2scalars(
    hdf_path: str, c_names: list[str], dataset: str | None = None, dtype: np.dtype = np.float32
) -> np.ndarray:
    """Extract scalar values from HDF5 file attributes."""
    with h5py.File(hdf_path, "r") as f:
        ds = f if dataset is None else f[dataset]
        l_array = [dtype(ds.attrs[n]).item() for n in c_names]
        scalars = np.stack(l_array, axis=-1)
        return scalars


class HDF_Attrs2Scalars(Transform):
    """Transform that extracts scalar values from HDF5 file attributes.

    Args:
        clm_names: attribute names to extract
        to_cls: callable to convert the resulting array to a specific type
    """

    def __init__(self, clm_names: list[str], to_cls: Callable = noop):
        self.clm_names = clm_names
        self.to_cls = to_cls

    def _extract_dict_scalars(self, item: dict | str | Path):
        match item:
            case dict():
                path = item["path"]
                dataset = item["dataset"] if "dataset" in item else None

                seq = hdf_attrs2scalars(path, self.clm_names, dataset)
            case str() | Path():
                seq = hdf_attrs2scalars(str(item), self.clm_names)
            case _:
                raise ValueError(f"Invalid item type: {type(item)}")
        return self.to_cls(seq)

    def encodes(self, item: dict | str | Path):
        return self._extract_dict_scalars(item)


def hdf_ds2scalars(
    hdf_path: str,
    clm_names: list[str],
    dataset: str | None = None,
    l_slc: int | None = None,
    r_slc: int | None = None,
    resampling_factor: float | None = None,
    fs_idx: int | None = None,
    dt_idx: int | None = None,
    fast_resample: bool = True,
    index: int | Callable | None = None,
    agg_func: Callable | None = None,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """Extract scalar values from HDF datasets using indexing or aggregation.

    Args:
        hdf_path: path to hdf5 file
        clm_names: list of dataset column names to extract
        dataset: dataset root for columns
        l_slc: left boundary for sequence window
        r_slc: right boundary for sequence window
        resampling_factor: scaling factor for sequence length
        fs_idx: index of frequency column for resampling
        dt_idx: index of time column for resampling
        fast_resample: use linear interpolation vs fft resampling
        index: specific index to extract or aggregation function
        agg_func: aggregation function to apply
        dtype: output data type for result array
    """
    seq = hdf_extract_sequence(
        hdf_path, clm_names, dataset, l_slc, r_slc, resampling_factor, fs_idx, dt_idx, fast_resample
    )
    if agg_func is not None and index is not None:
        raise ValueError("Cannot specify both agg_func and index. Choose one.")
    elif agg_func is not None:
        result = agg_func(seq, axis=0)
    elif index is not None:
        if callable(index):
            result = index(seq, axis=0)
        else:
            result = seq[index]
    else:
        result = seq[-1]
    return result.astype(dtype)


class HDF_DS2Scalars(Transform):
    """Extract scalar values from HDF datasets using indexing or aggregation.

    Args:
        clm_names: list of dataset column names to extract
        index: specific index to extract or aggregation function
        agg_func: aggregation function to apply
        to_cls: transform to apply to final result
    """

    def __init__(
        self,
        clm_names: list[str],
        index: int | Callable | None = None,
        agg_func: Callable | None = None,
        to_cls: Callable = noop,
        **extract_kwargs,
    ):
        self.clm_names = clm_names
        self.index = index
        self.agg_func = agg_func
        self.to_cls = to_cls
        self.extract_kwargs = extract_kwargs

    def _extract_dict_scalars(self, item: dict | str | Path):
        match item:
            case dict():
                path = item["path"]
                dataset = item.get("dataset", self.extract_kwargs.get("dataset"))
                l_slc = item.get("l_slc")
                r_slc = item.get("r_slc")
                resampling_factor = item.get("resampling_factor")
                kwargs = {**self.extract_kwargs}
                if dataset is not None:
                    kwargs["dataset"] = dataset
                if l_slc is not None:
                    kwargs["l_slc"] = l_slc
                if r_slc is not None:
                    kwargs["r_slc"] = r_slc
                if resampling_factor is not None:
                    kwargs["resampling_factor"] = resampling_factor
                result = hdf_ds2scalars(path, self.clm_names, index=self.index, agg_func=self.agg_func, **kwargs)
            case str() | Path():
                result = hdf_ds2scalars(
                    str(item), self.clm_names, index=self.index, agg_func=self.agg_func, **self.extract_kwargs
                )
            case _:
                raise ValueError(f"Invalid item type: {type(item)}")
        return self.to_cls(result)

    def encodes(self, item: dict | str | Path):
        return self._extract_dict_scalars(item)


class TensorSequences(TensorBase):
    """Tensor type for time series sequence data with plotting support."""

    #     def __init__(self,x,c_names=None, **kwargs):
    #         super().__init__()
    #         self.c_names = c_names

    def show(self, ctx: plt.Axes | None = None, **kwargs) -> plt.Axes:
        """Display the sequence as a line plot."""
        # Get the figure and axis
        if ctx is None:
            fig, ax = plt.subplots(figsize=kwargs.get("figsize", (8, 4)))
        else:
            ax = ctx

        # Plot the sequence as a line
        ax.plot(self.cpu().numpy(), **kwargs)
        ax.grid(True, alpha=0.3)

        if ctx is None:
            plt.tight_layout()
        return ax

    @classmethod
    @delegates(HDF2Sequence, keep=True)
    def from_hdf(cls, clm_names: list[str], **kwargs) -> HDF2Sequence:
        """Create a transform for extracting sequences from HDF5 files."""
        return HDF2Sequence(clm_names, **kwargs)


class TensorSequencesInput(TensorSequences):
    """Input tensor type for time series sequences."""

    pass


class TensorSequencesOutput(TensorSequences):
    """Output tensor type for time series sequences."""

    pass


@Transform
def toTensorSequencesInput(o) -> TensorSequencesInput:
    """Convert input to TensorSequencesInput."""
    return TensorSequencesInput(o)


@Transform
def toTensorSequencesOutput(o) -> TensorSequencesOutput:
    """Convert input to TensorSequencesOutput."""
    return TensorSequencesOutput(o)


class TensorScalars(TensorBase):
    """Tensor type for scalar values with formatting and plotting support."""

    def __format__(
        self,
        format_spec: str,
    ) -> str:
        """Format tensor scalars using standard format specifications."""
        if self.ndim == 0:
            # 0-dimensional tensor (single scalar)
            return format(self.item(), format_spec)
        else:
            # 1-dimensional tensor (multiple scalars)
            if format_spec:
                formatted_values = [format(val.item(), format_spec) for val in self]
                return f"[{', '.join(formatted_values)}]"
            else:
                return str(self.cpu().numpy())

    def show(
        self,
        ctx: plt.Axes | None = None,
        labels: list[str] | None = None,
        title_prefix: str = "",
        format_spec: str = ".3g",
        **kwargs,
    ) -> plt.Axes:
        """Show scalar values as plot title with optional labels.

        Args:
            ctx: matplotlib axes to draw on
            labels: labels for each scalar value
            title_prefix: prefix for plot title
            format_spec: format specification for values
        """
        if ctx is None:
            figsize = kwargs.pop("figsize", None)
            ctx = plt.subplots(figsize=figsize)[1]
            ctx.axis("off")

        values = [self] if self.ndim == 0 else list(self)
        formatted_parts = []

        for i, val in enumerate(values):
            val_str = format(val.item(), format_spec)
            label = labels[i] if labels and i < len(labels) else None
            formatted_parts.append(f"{label}: {val_str}" if label else val_str)

        title = ", ".join(formatted_parts)
        if title_prefix:
            title = f"{title_prefix}: {title}"

        ctx.set_title(title, **kwargs)
        return ctx

    @classmethod
    @delegates(HDF_Attrs2Scalars, keep=True)
    def from_hdf_attrs(
        cls,
        clm_names: list[str],
        **kwargs,
    ) -> HDF_Attrs2Scalars:
        """Create a transform for extracting scalars from HDF attributes."""
        return HDF_Attrs2Scalars(clm_names, **kwargs)

    @classmethod
    @delegates(HDF_DS2Scalars, keep=True)
    def from_hdf_ds(
        cls,
        clm_names: list[str],
        **kwargs,
    ) -> HDF_DS2Scalars:
        """Create a transform for extracting scalars from HDF datasets."""
        return HDF_DS2Scalars(clm_names, **kwargs)


class TensorScalarsInput(TensorScalars):
    """Input tensor type for scalar values."""

    pass


class TensorScalarsOutput(TensorScalars):
    """Output tensor type for scalar values."""

    pass


@Transform
def toTensorScalarsInput(o) -> TensorScalarsInput:
    """Convert input to TensorScalarsInput."""
    return TensorScalarsInput(o)


@Transform
def toTensorScalarsOutput(o) -> TensorScalarsOutput:
    """Convert input to TensorScalarsOutput."""
    return TensorScalarsOutput(o)


for f in (
    torch.nn.functional.mse_loss,
    torch.nn.functional.huber_loss,
    Tensor.__getitem__,
    Tensor.__ne__,
    Tensor.__eq__,
    Tensor.add,
    Tensor.sub,
    Tensor.mul,
    Tensor.div,
    Tensor.__rsub__,
    Tensor.__radd__,
    Tensor.matmul,
    Tensor.bmm,
):
    TensorBase.register_func(f, TensorSequences)
    TensorBase.register_func(f, TensorScalars)


def plot_sequence(axs: list, in_sig: Tensor, targ_sig: Tensor, out_sig: Tensor | None = None, **kwargs):
    """Plot input, target, and optional prediction sequences on subplot axes."""
    n_targ = targ_sig.shape[1]  # Number of supervised outputs
    n_out = out_sig.shape[1] if out_sig is not None else n_targ  # Total model outputs
    n_ax = len(axs) - 1  # Axes available for outputs (last one for input)

    # Plot all output channels (supervised + auxiliary)
    for j in range(min(n_ax, max(n_targ, n_out))):
        ax = axs[j]
        # Plot target if available (only for supervised channels)
        if j < n_targ:
            ax.plot(targ_sig[:, j], label="y", alpha=0.7)
        # Plot prediction if available
        if out_sig is not None and j < n_out:
            label = "ŷ" if j < n_targ else "ŷ (aux)"
            ax.plot(out_sig[:, j], label=label, alpha=0.7)
        # Plot reference if provided
        if "ref" in kwargs and j < kwargs["ref"].shape[1]:
            ax.plot(kwargs["ref"][:, j], label="ref", alpha=0.5)
        # Mark auxiliary channels
        if j >= n_targ:
            ax.set_title(f"Channel {j} (auxiliary)", fontsize=10)
        ax.legend(fontsize=8)
        ax.label_outer()
    axs[-1].plot(in_sig)


def plot_seqs_single_figure(
    n_samples: int, n_targ: int, samples: list, plot_func: Callable, outs: list | None = None, **kwargs
):
    """Plot multiple sample sequences in a single figure grid."""
    rows = max(1, ((n_samples - 1) // 3) + 1)
    cols = min(3, n_samples)
    fig = plt.figure(figsize=(9, 2 * cols))
    outer_grid = fig.add_gridspec(rows, cols)
    #     import pdb; pdb.set_trace()
    for i in range(n_samples):
        in_sig = samples[i][0]
        targ_sig = samples[i][1]
        if outs is not None:
            out_sig = outs[i][0]
        inner_grid = outer_grid[i].subgridspec(n_targ + 1, 1)
        axs = [fig.add_subplot(inner_grid[j]) for j in range(n_targ + 1)]
        plot_func(axs, in_sig, targ_sig, out_sig=out_sig if outs is not None else None, **kwargs)
    plt.tight_layout()


def plot_seqs_multi_figures(
    n_samples: int, n_targ: int, samples: list, plot_func: Callable, outs: list | None = None, **kwargs
):
    """Plot each sample sequence in its own separate figure."""
    for i in range(n_samples):
        fig = plt.figure(figsize=(9, 3))
        axs = fig.subplots(nrows=n_targ + 1, sharex=True)
        in_sig = samples[i][0]
        targ_sig = samples[i][1]
        if outs is not None:
            out_sig = outs[i][0]

        plot_func(axs, in_sig, targ_sig, out_sig=out_sig if outs is not None else None, **kwargs)

        plt.tight_layout()


from plum import dispatch


@dispatch
def show_batch(x: TensorSequences, y: TensorSequences, samples: list, ctxs=None, max_n: int = 6, **kwargs):
    """Display a batch of input/target sequence pairs."""
    n_samples = min(len(samples), max_n)
    n_targ = samples[0][1].shape[1]
    if n_samples > 3:
        # if there are more then 3 samples to plot then put them in a single figure
        plot_seqs_single_figure(n_samples, n_targ, samples, plot_sequence, **kwargs)
    else:
        # if there are less then 3 samples to plot then put each in its own figure
        plot_seqs_multi_figures(n_samples, n_targ, samples, plot_sequence, **kwargs)
    return ctxs


@dispatch
def show_results(
    x: TensorSequences, y: TensorSequences, samples: list, outs: list, ctxs=None, max_n: int = 2, **kwargs
):
    """Display input/target/prediction results for model evaluation."""
    n_samples = min(len(samples), max_n)
    n_targ = samples[0][1].shape[1]
    if n_samples > 3:
        # if there are more then 3 samples to plot then put them in a single figure
        plot_seqs_single_figure(n_samples, n_targ, samples, plot_sequence, outs, **kwargs)
    else:
        # if there are less then 3 samples to plot then put each in its own figure
        plot_seqs_multi_figures(n_samples, n_targ, samples, plot_sequence, outs, **kwargs)
    return ctxs
