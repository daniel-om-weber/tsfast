
__all__ = ['mse_nan', 'ignore_nan', 'float64_func', 'SkipNLoss', 'CutLoss', 'weighted_mae', 'RandSeqLenLoss', 'fun_rmse',
           'cos_sim_loss', 'cos_sim_loss_pow', 'nrmse', 'nrmse_std', 'mean_vaf', 'zero_loss']

from ..data import *
from fastai.basics import *
import warnings

import functools

def ignore_nan(func):
    '''remove nan values from tensors before function execution, reduces tensor to a flat array, apply to functions such as mse'''
    @functools.wraps(func)
    def ignore_nan_decorator(*args, **kwargs):
#         mask = ~torch.isnan(args[-1]) #nan mask of target tensor
#         args = tuple([x[mask] for x in args]) #remove nan values
        mask = ~torch.isnan(args[-1][...,-1]) #nan mask of target tensor
        args = tuple([x[mask,:] for x in args]) #remove nan values
        return func(*args, **kwargs)
    return ignore_nan_decorator

mse_nan = ignore_nan(mse)

import functools
import warnings

def float64_func(func):
    '''calculate function internally with float64 and convert the result back'''
    @functools.wraps(func)
    def float64_func_decorator(*args, **kwargs):
        typ = args[0].dtype
        try:
            # Try to use float64 for higher precision
            args = tuple([x.double() if issubclass(type(x),Tensor) else x for x in args])
            return func(*args, **kwargs).type(typ)
        except TypeError as e:
            # If float64 is not supported on this device, warn the user and fall back to float32
            if "doesn't support float64" in str(e):
                warnings.warn(f"Float64 precision not supported on {args[0].device} device. Using original precision. This may reduce numerical accuracy. Error: {e}")
                return func(*args, **kwargs)
            else:
                raise # Re-raise if it's some other error
    return float64_func_decorator

def SkipNLoss(fn,n_skip=0):
    '''Loss-Function modifier that skips the first n samples of sequential data'''
    @functools.wraps(fn)
    def _inner( input, target):
        return fn(input[:,n_skip:].contiguous(),target[:,n_skip:].contiguous())
    
    return _inner

def CutLoss(fn,l_cut=0,r_cut=None):
    '''Loss-Function modifier that skips the first n samples of sequential data'''
    @functools.wraps(fn)
    def _inner( input, target):
        return fn(input[:,l_cut:r_cut],target[:,l_cut:r_cut])
    
    return _inner

def weighted_mae(input, target):
    max_weight = 1.0
    min_weight = 0.1
    seq_len = input.shape[1]

    device = input.device
    if device.type == 'mps':
        # Compute on CPU because MPS does not support logspace yet
        weights = torch.logspace(start=torch.log10(torch.tensor(max_weight)),
                                end=torch.log10(torch.tensor(min_weight)),
                                steps=seq_len, device='cpu').to(device)
        warnings.warn(f"torch.logspace not supported on {device} device. Using cpu. This may reduce numerical performance")
    else:
        # Compute directly on the target device 
        weights = torch.logspace(start=torch.log10(torch.tensor(max_weight)),
                                end=torch.log10(torch.tensor(min_weight)),
                                steps=seq_len, device=device)


    weights = (weights / weights.sum())[None,:,None]

    return ((input-target).abs()*weights).sum(dim=1).mean()


def RandSeqLenLoss(fn,min_idx=1,max_idx=None,mid_idx=None):
    '''Loss-Function modifier that truncates the sequence length of every sequence in the minibatch inidiviually randomly.
    At the moment slow for very big batchsizes.'''
    @functools.wraps(fn)
    def _inner( input, target):
        bs,seq_len,_ = input.shape
        if 'max_idx' not in locals():  max_idx = seq_len
        if 'mid_idx' not in locals():  mid_idx = min_idx#+(max_idx-min_idx)//4
        # len_list = torch.randint(min_idx,max_idx,(bs,))
        len_list = np.random.triangular(min_idx,mid_idx,max_idx,(bs,)).astype(int)
        return torch.stack([fn(input[i,:len_list[i]],target[i,:len_list[i]]) for i in range(bs)]).mean()
    return _inner

def fun_rmse(inp, targ): 
    '''rmse loss function defined as a function not as a AccumMetric'''
    return torch.sqrt(F.mse_loss(inp, targ))

def cos_sim_loss(inp, targ): 
    '''rmse loss function defined as a function not as a AccumMetric'''
    return (1-F.cosine_similarity(inp,targ,dim=-1)).mean()

def cos_sim_loss_pow(inp, targ): 
    '''rmse loss function defined as a function not as a AccumMetric'''
    return (1-F.cosine_similarity(inp,targ,dim=-1)).pow(2).mean()

def nrmse(inp, targ): 
    '''rmse loss function scaled by variance of each target variable'''
    mse = (inp-targ).pow(2).mean(dim=[0,1])
    var = targ.var(dim=[0,1])
    return (mse/var).sqrt().mean()

def nrmse_std(inp, targ): 
    '''rmse loss function scaled by standard deviation of each target variable'''
    mse = (inp-targ).pow(2).mean(dim=[0,1])
    var = targ.std(dim=[0,1])
    return (mse/var).sqrt().mean()

def mean_vaf(inp,targ):
    return (1-((targ-inp).var()/targ.var()))*100

def zero_loss(pred, targ):
    return (pred * 0).sum()
