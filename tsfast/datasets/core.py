
__all__ = ['create_dls_test', 'extract_mean_std_from_dls', 'dict_file_save', 'dict_file_load', 'extract_mean_std_from_hdffiles',
           'extract_mean_std_from_dataset', 'is_dataset_directory', 'create_dls', 'get_default_dataset_path',
           'get_dataset_path', 'clean_default_dataset_path', 'create_dls_downl', 'estimate_norm_stats',
           'NormPair', 'NormStats']

from fastai.data.all import *
from ..data import *

from ..data.loader import *
import h5py
from pathlib import Path as _Path
from shutil import rmtree
import warnings
from dataclasses import dataclass

@dataclass
class NormPair:
    'Per-signal normalization statistics (mean, std, min, max as 1-D numpy arrays).'
    mean: np.ndarray
    std: np.ndarray
    min: np.ndarray
    max: np.ndarray

    def __add__(self, other):
        'Concatenate two NormPairs feature-wise (e.g. norm_u + norm_y).'
        return NormPair(*(np.hstack([a, b]) for a, b in zip(self, other)))

    def __iter__(self):
        return iter((self.mean, self.std, self.min, self.max))

    def __getitem__(self, idx):
        return (self.mean, self.std, self.min, self.max)[idx]

from typing import NamedTuple

class NormStats(NamedTuple):
    u: NormPair
    x: NormPair | None
    y: NormPair

def extract_mean_std_from_dls(dls):
    'Extract normalization statistics stored on the DataLoaders object.'
    if not hasattr(dls, 'norm_stats'): raise AttributeError("DataLoaders missing norm_stats. Use create_dls to create them.")
    return dls.norm_stats

def dict_file_save(key,value,f_path='dls_normalize.p'):
    'save value to a dictionary file, appends if it already exists'
    
    #use the absolute path, so seperate processes refer to the same file
    f_path = Path(f_path)
    try:
        with open( f_path, "rb" ) as f:
            dictionary = pickle.load( f )
    except OSError:
        dictionary = {}
    dictionary[key] = value
        
    with open( f_path, "wb" ) as f:
        pickle.dump(dictionary,f)

def dict_file_load(key,f_path='dls_normalize.p'):
    'load value from a dictionary file'

    #use the absolute path, so seperate processes refer to the same file
    f_path = Path(f_path)
    try:
        with open( f_path, "rb" ) as f:
            dictionary = pickle.load( f )
        if key in dictionary:
            return dictionary[key]
    except OSError:
        print(f'{f_path} not found')
    return None

def extract_mean_std_from_hdffiles(
        lst_files, #List of paths to HDF5 files
        lst_signals #List of signal names, each a dataset within the HDF5 files
        ):
    'Calculate the mean and standard deviation of the signals from the provided HDF5 files.'
    if len(lst_signals) == 0: return (None,None)

    # Initialize accumulators for mean and squared mean
    sums = np.zeros(len(lst_signals))
    squares = np.zeros(len(lst_signals))
    counts = 0
    
    # Process each file
    for file in lst_files:
        with h5py.File(file, 'r') as f:
            for i, signal in enumerate(lst_signals):
                data = f[signal][:]
                if data.ndim > 1: raise ValueError(f'Each dataset in a file has to be 1d. {signal} is {data.ndim}.')
                sums[i] += np.sum(data)
                squares[i] += np.sum(data ** 2)
                
        counts += data.size
    
    # Calculate the mean and standard deviation across all files
    means = sums / counts
    stds = np.sqrt((squares / counts) - (means ** 2))
    
    return means.astype(np.float32), stds.astype(np.float32)

def extract_mean_std_from_dataset(lst_files,u,x,y):
    train_files = lst_files[ParentSplitter()(lst_files)[0]]
    norm_u = extract_mean_std_from_hdffiles(train_files,u)
    norm_x = extract_mean_std_from_hdffiles(train_files,x)
    norm_y = extract_mean_std_from_hdffiles(train_files,y)
    return (norm_u,norm_x,norm_y)

def estimate_norm_stats(dls, n_batches:int=10):
    'Estimate per-feature mean/std/min/max from training batches. Returns tuple of NormPair, one per batch element.'
    acc = None
    for i, batch in enumerate(dls.train):
        if i >= n_batches: break
        if acc is None: acc = [[t] for t in batch]
        else:
            for j, t in enumerate(batch): acc[j].append(t)

    stats = []
    for tensors in acc:
        t = torch.cat(tensors).flatten(0, -2)  # [total_samples, features]
        stats.append(NormPair(
            mean=t.mean(0).numpy().astype(np.float32),
            std=t.std(0).numpy().astype(np.float32),
            min=t.min(0).values.numpy().astype(np.float32),
            max=t.max(0).values.numpy().astype(np.float32)))
    return tuple(stats)

def is_dataset_directory(ds_path):
    """
    Checks if the given directory path is a dataset with hdf5 files.
    
    :param ds_path: The path to the directory to check.
    :return: True if the directory contains 'train', 'valid', and 'test' subdirectories,
             each of which must contain at least one HDF5 file. False otherwise.
    """
    required_dirs = ['train', 'valid', 'test']
    
    # Check if all required directories exist and contain HDF5 files
    for dir_name in required_dirs:
        dir_path = os.path.join(ds_path, dir_name)
        if not os.path.exists(dir_path) or not os.path.isdir(dir_path):
            return False  # Directory missing
        hdf_files = get_hdf_files(dir_path, recurse=True)
        if not hdf_files:
            return False  # No HDF5 files found in this directory

    return True  # All checks passed

def create_dls(
        u, #list of input signal names
        y, #list of output signal names
        dataset:Path|list, #path to dataset with train,valid and test folders, or list of filepaths
        win_sz:int = 100, #initial window size
        x:list =[], #optional list of state signal names
        stp_sz:int = 1, #step size between consecutive windows
        sub_seq_len:int = None, #if provided uses truncated backpropagation throug time with this sub sequence length 
        bs:int = 64, #batch size
        prediction:bool = False, #if true, the output is concatenated to the input, mainly for prediction tasks
        input_delay:bool = False, #if true, the input is delayed by one step
        valid_stp_sz:int = None, #step size between consecutive validation windows, defaults to win_sz
        cached:bool = True, #if true, the data is cached in RAM
        num_workers:int = 5, #number of processes for the dataloader, 0 for no multiprocessing
        n_batches_train:int|None = 300, #exact number of training batches per epoch
        n_batches_valid:int|None = None, #exact number of validation batches per epoch
        max_batches_training:int|None = None, #DEPRECATED: limits the number of training batches in a single epoch
        max_batches_valid:int|None = None, #DEPRECATED: limits the number of validation batches in a single epoch
    ):
    if valid_stp_sz is None: valid_stp_sz = win_sz

    #extract list of dataset files
    if isinstance(dataset, (Path,str)):
        hdf_files = get_hdf_files(dataset)
    elif isinstance(dataset,(list,tuple,L)):
        hdf_files = dataset
    else:
        raise ValueError(f'dataset has to be a Path or filelist. {type(dataset)} was given.')
    
    #choose input and output signal blocks
    if prediction:
        if input_delay: #if true, the input is delayed by one step
            blocks = (SequenceBlock.from_hdf(u+x+y,TensorSequencesInput,clm_shift=[-1]*len(u+x+y),cached=cached),
                        SequenceBlock.from_hdf(y,TensorSequencesOutput,clm_shift=[1]*len(y),cached=cached))
        else:
            blocks = (SequenceBlock.from_hdf(u+x+y,TensorSequencesInput,clm_shift=([0]*len(u)+[-1]*len(x+y)),cached=cached),
                        SequenceBlock.from_hdf(y,TensorSequencesOutput,clm_shift=[1]*len(y),cached=cached))
    else:
        blocks = (SequenceBlock.from_hdf(u,TensorSequencesInput,cached=cached),
                SequenceBlock.from_hdf(y,TensorSequencesOutput,cached=cached))


    seq = DataBlock(blocks=blocks,
                     get_items=CreateDict([DfApplyFuncSplit(
                            ParentSplitter(),
                            DfHDFCreateWindows(win_sz=win_sz,stp_sz=stp_sz,clm=u[0]),
                            DfHDFCreateWindows(win_sz=win_sz,stp_sz=valid_stp_sz,clm=u[0])
                        )]),
                     splitter=ParentSplitter())
    
    # Determine which factory to use based on parameters
    use_old_factory = False
    if max_batches_training is not None or max_batches_valid is not None:
        # Old parameters provided
        if n_batches_train is not None or n_batches_valid is not None:
            # Both old and new provided - new takes precedence
            warnings.warn(
                "Both old ('max_batches_training', 'max_batches_valid') and new ('n_batches_train', 'n_batches_valid') "
                "parameters provided. Using new parameters. Old parameters are deprecated.",
                DeprecationWarning,
                stacklevel=2
            )
        else:
            # Only old parameters - use old factory with warning
            warnings.warn(
                "Parameters 'max_batches_training' and 'max_batches_valid' are deprecated. "
                "Use 'n_batches_train' and 'n_batches_valid' instead.",
                DeprecationWarning,
                stacklevel=2
            )
            use_old_factory = True
    
    if use_old_factory:
        # Use deprecated BatchLimit_Factory for backwards compatibility
        if sub_seq_len is None:
            dl_kwargs=[{'max_batches':max_batches_training},
                       {'max_batches':max_batches_valid}]
            dl_type = BatchLimit_Factory(TfmdDL)
        else:
            dl_kwargs=[{'sub_seq_len':sub_seq_len,'max_batches':max_batches_training},
                       {'sub_seq_len':sub_seq_len,'max_batches':max_batches_valid}]
            dl_type = BatchLimit_Factory(TbpttDl)
    else:
        # Use new NBatches_Factory
        if sub_seq_len is None:
            dl_kwargs=[{'n_batches':n_batches_train},
                       {'n_batches':n_batches_valid}]
            dl_type = NBatches_Factory(TfmdDL)
        else:
            dl_kwargs=[{'sub_seq_len':sub_seq_len,'n_batches':n_batches_train},
                       {'sub_seq_len':sub_seq_len,'n_batches':n_batches_valid}]
            dl_type = NBatches_Factory(TbpttDl)
        
    dls = seq.dataloaders(hdf_files,bs=bs,num_workers=num_workers,
                          dl_type=dl_type,dl_kwargs=dl_kwargs)

    #estimate normalization stats from training batches
    input_stats, output_stats = estimate_norm_stats(dls)
    n_u, n_x = len(u), len(x)
    norm_u = NormPair(input_stats.mean[:n_u], input_stats.std[:n_u],
                      input_stats.min[:n_u], input_stats.max[:n_u])
    norm_x = NormPair(input_stats.mean[n_u:n_u+n_x], input_stats.std[n_u:n_u+n_x],
                      input_stats.min[n_u:n_u+n_x], input_stats.max[n_u:n_u+n_x]) if n_x > 0 else None
    norm_y = output_stats
    dls.norm_stats = NormStats(norm_u, norm_x, norm_y)

    #add the test dataloader
    test_hdf_files =  hdf_files.filter(lambda o:Path(o).parent.name == 'test')
    if prediction:
        items = CreateDict([DfHDFCreateWindows(win_sz=win_sz,stp_sz=win_sz,clm=u[0])])(test_hdf_files)
        test_dl = dls.test_dl(items,bs=min(bs,len(items)), with_labels=True)
    else:
        items = CreateDict()(test_hdf_files)
        test_dl = dls.test_dl(items,bs=1, with_labels=True)

    dls.loaders.append(test_dl)
    
    return dls

def _get_project_root():
    """Walk up from this file to find the project root (directory containing test_data/)."""
    d = _Path(__file__).resolve().parent
    while d != d.parent:
        if (d / 'test_data').is_dir(): return d
        d = d.parent
    raise FileNotFoundError("Could not find project root containing test_data/")

create_dls_test = partial(
        create_dls,
        u=['u'],y=['y'],
        dataset=_get_project_root() / 'test_data/WienerHammerstein',
        win_sz=100,
        stp_sz=100
    )
create_dls_test.__doc__ = 'create a dataloader from a small dataset provided with tsfast'

def get_default_dataset_path():
    "Create a directory in the user's home directory for storing datasets"
    data_dir = Path.home() / '.tsfast' / 'datasets'
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir

def get_dataset_path():
    "Retrieves the tsfast dataset directory. Tries to read the path in the environment variable 'TSFAST_PATH', returns the default otherwise."
    env_var_name = 'TSFAST_PATH'
    env_path = os.getenv(env_var_name)

    if env_path:
        return Path(env_path)
    else:
        return get_default_dataset_path()

def clean_default_dataset_path():
    "Removes the default directory where the datasets are stored"
    rmtree(get_default_dataset_path())

@delegates(create_dls, keep=True)
def create_dls_downl(
    dataset=None,#path to the dataset directory, if not provided uses default
    download_function=None,# function 
    **kwargs
):
    if dataset is None and download_function is not None:
        dataset = get_dataset_path() / download_function.__name__
    else:
        dataset = Path(dataset)

    if not is_dataset_directory(dataset):
        if download_function is not None:
            print(f'Dataset not found. Downloading it to "{dataset}"')
            download_function(dataset)
        else:
            raise ValueError(f'{dataset} does not contain a dataset. Check the path or activate the download flag.')

    return create_dls(dataset=dataset,**kwargs)
