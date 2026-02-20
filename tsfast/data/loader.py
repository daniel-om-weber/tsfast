
__all__ = ['TbpttDl', 'reset_model_state', 'TbpttResetCB', 'WeightedDL_Factory', 'uniform_p_of_category', 'uniform_p_of_float',
           'uniform_p_of_float_with_gaps', 'BatchLimit_Factory', 'NBatches_Factory', 'get_inp_out_size']

from fastai.basics import *
from .core import *

from torch.utils.data.dataloader import _MultiProcessingDataLoaderIter,_SingleProcessDataLoaderIter
_loaders = (_MultiProcessingDataLoaderIter,_SingleProcessDataLoaderIter)

@delegates()
class TbpttDl(TfmdDL):

    def __init__(self, dataset, sub_seq_len=None, seq_len = None ,shuffle=True,num_workers=2, **kwargs):
#         assert sub_seq_len is not None
        store_attr('sub_seq_len,seq_len')
        self.rnn_reset = False
        super().__init__(dataset=dataset, shuffle=shuffle, num_workers=num_workers, **kwargs)
        # check for divisibility
        if self.sub_seq_len is not None:
            if self.seq_len is None:
                if len(self.dataset) == 0:
                    raise ValueError("Cannot determine seq_len from an empty dataset.")
                self.seq_len = self.do_item(0)[0].shape[0]

            if self.seq_len % self.sub_seq_len != 0:
                warnings.warn(
                    f"Sequence length ({self.seq_len}) is not perfectly divisible by sub_seq_len ({self.sub_seq_len}). "
                    f"The last segment of each sequence in TbpttDl will be shorter.", UserWarning
                )
        
    @property
    def n_sub_seq(self):
        if self.sub_seq_len is None: 
            return 1
        if self.seq_len is None: 
            self.seq_len = self.do_item(0)[0].shape[0]
        return math.ceil(self.seq_len / self.sub_seq_len)
        
    def __len__(self):
        return super().__len__() * self.n_sub_seq
    
    def _next_worker(self,w_id):
        w_id += 1
        if w_id > self.fake_l.num_workers-1: 
            w_id = 0
        return w_id

    def sample(self):
        #replaced new fastai sample formulation that store __idxs in main process
        return (b for i,b in enumerate(self.__idxs) if i//(self.bs or 1)%self.num_workers==self.offs)
#         return (b for i,b in enumerate(self.get_idxs()) if i//(self.bs or 1)%self.num_workers==self.offs)
            
    def __iter__(self):
        '''iterator that handles multiprocessing by caching samples that are generated out of order'''
        self.randomize()
        self.before_iter()
        self.__idxs=self.get_idxs() # called in context of main process (not workers/subprocesses)
        
        queue = {n:[] for n in range(self.fake_l.num_workers)} 
        current_worker = None
        idx = 0
        max_batches = len(self)

        for loaded_b,w_id in _loaders[self.fake_l.num_workers==0](self.fake_l):            
            if idx >= max_batches:
                break

            if w_id is None:
                self.rnn_reset=True
                b= loaded_b
                self.rnn_reset = (idx % self.n_sub_seq) == 0
                yield self.after_batch(b if self.device is None else to_device(b, self.device))
                idx += 1 #idx increments after every yield, not every loop
            else:
                if current_worker is None:
                    current_worker = w_id
                
                #retrieve queued elements from worker
                while len(queue[current_worker]) > 0:
                    b = queue[current_worker].pop(0)
                    self.rnn_reset = (idx % self.n_sub_seq) == 0
                    yield self.after_batch(b if self.device is None else to_device(b, self.device))
                    idx += 1
                    if (idx % self.n_sub_seq) == 0:
                        current_worker = self._next_worker(current_worker) #next worker, stay in loop for the queue
                        
                
                #retrieve fresh elements from worker
                if w_id != current_worker: #not active worker
                    queue[w_id] += [loaded_b]
                    continue
                else:#active worker
                    b = loaded_b
                    self.rnn_reset = (idx % self.n_sub_seq) == 0
                    yield self.after_batch(b if self.device is None else to_device(b, self.device))
                    idx += 1 #idx increments after every yield, not every loop
                    if (idx % self.n_sub_seq) == 0:
                        current_worker = self._next_worker(current_worker)
                
        self.after_iter()
        if hasattr(self, 'it'): 
            del(self.it)
    
    def create_batches(self, samps):
        yield from self._tbptt_generator(super().create_batches(samps))
        
    def _tbptt_generator(self,batch_iter):
        '''generator function that splits batches in smaller windows, yields mini_batch and worker id'''
        for b in batch_iter:
            for i in range(self.n_sub_seq):
                #it is importan to retain the tuple type, or future transforms may not work
                if self.sub_seq_len is None:
                    trunc_b = b
                else:
                    trunc_b = tuple([retain_type(x[:,i*self.sub_seq_len:(i+1)*self.sub_seq_len],x) for x in b])
                yield trunc_b, (None if torch.utils.data.get_worker_info() is None else torch.utils.data.get_worker_info().id)
        

def reset_model_state(model):
    for m in model.modules():
        if hasattr(m,'reset_state'): 
            m.reset_state()

class TbpttResetCB(Callback):
    "`Callback` resets the rnn model with every new sequence for tbptt, calls `reset_state` in every module of the model"
        
    def before_batch(self):
        dl = self.learn.dls.train if self.training else self.learn.dls.valid
#         if not self.training: import pdb; pdb.set_trace()
        if (hasattr(dl,'rnn_reset') and dl.rnn_reset) or not hasattr(dl,'rnn_reset'):
            reset_model_state(self.learn.model)
        
    def after_fit(self): 
        reset_model_state(self.learn.model)

def WeightedDL_Factory(cls):
    '''
    Weighted Dataloader that provides control over sampling probabilities.
    wgts: probability array with probability for every item
            gets extracted from the pandas 'p_sample' column if given. 
            Otherwise uniform sampling will be enabled
        
    '''
    assert issubclass(cls, TfmdDL)
    
    class WeightedDL(cls):
        def __init__(self, dataset, wgts=None, **kwargs):
#             import pdb;pdb.set_trace()
            self.wgts = None
            #self.items need to be assigned, but super.init needs wgts allready assigned
            super().__init__(dataset=dataset, **kwargs) 
            if wgts is None:
                if (isinstance(self.items,pd.DataFrame) and
                    len(self.items) > 0 and 
                    'p_sample' in self.items):
                    self.wgts = self.items.p_sample.to_numpy()
                    self.wgts = self.wgts/self.wgts.sum()
                elif (isinstance(self.items,Iterable) and
                    len(self.items) > 0 and 
                    hasattr(self.items[0],'keys') and 
                    'p_sample' in self.items[0].keys()):
                    self.wgts = np.array([x['p_sample'] for x in self.items])
                    self.wgts = self.wgts/self.wgts.sum()
                else:
                    print('No wgts provided for WeightedDL. Was that intentional?')
            else:
                self.wgts = wgts/np.sum(wgts)

        def get_idxs(self):
            if self.n==0: 
                return []
            if not self.shuffle or self.wgts is None: 
                return super().get_idxs()
            #calculate number of elements with length of the dataset, for batch truncation
            idxs = list(np.random.choice(self.n, size=len(self)*self.bs, p=self.wgts))
            return idxs
    return WeightedDL

def uniform_p_of_category(cat_name):  
    '''Scales sampling weights for an even distribution between every category'''
    def _inner(df):
        if 'p_sample' in df:
            df_targ = df.drop('p_sample',axis='columns')
        else:
            df_targ = df
            
        counts = df_targ[cat_name].value_counts()
        sample_prob =  1/counts
        sample_prob.name = 'p_sample'
        df_res = df_targ.merge(sample_prob,left_on=cat_name,right_index=True)
        
        if 'p_sample' in df: 
            df_res.p_sample = df_res.p_sample* df.p_sample.values
            
        df_res.p_sample /= df_res.p_sample.sum()
            
        return df_res
    
    return _inner

def uniform_p_of_float(var_name,bins = 10):
    '''Scales sampling weights for an even distribution of the continous variable by creating equi sized bins'''
    def _inner(df):
        if 'p_sample' in df:
            df_targ = df.drop('p_sample',axis='columns')
        else:
            df_targ = df
            
        df_targ['bins'] = pd.cut(df_targ[var_name], bins)
        counts = df_targ['bins'].value_counts()
        sample_prob =  1/counts
        sample_prob.name = 'p_sample'
        df_res = df_targ.merge(sample_prob,left_on='bins',right_index=True)
        df_res.drop(['bins'],axis='columns',inplace=True)
        
        if 'p_sample' in df: 
            df_res.p_sample = df_res.p_sample* df.p_sample.values
            
        df_res.p_sample /= df_res.p_sample.sum()
        
        return df_res

    return _inner

def uniform_p_of_float_with_gaps(var_name,bins = 100):
    '''Scales sampling weights for an even distribution of the continous variable by creating equi sized bins'''
    def _inner(df):
        if 'p_sample' in df:
            df_targ = df.drop('p_sample',axis='columns')
        else:
            df_targ = df
            
        length = df_targ[var_name].max()-df_targ[var_name].min() #value range
        df_targ['bins'] = pd.qcut(df_targ[var_name],bins,duplicates='drop') #bins with rougly the same size
        df_targ['p_sample'] =  df_targ['bins'].apply(lambda x: x.length).astype('f8')/length #sample_prob by bin width
        sample_prob =  1/df_targ['bins'].value_counts() #correct uneven bin distribution
        sample_prob.name = 'p_sample_correction'
        df_res = df_targ.merge(sample_prob,left_on='bins',right_index=True)
        
        df_res.p_sample *= df_res.p_sample_correction
        df_res.drop(['bins','p_sample_correction'],axis='columns',inplace=True)

        if 'p_sample' in df: 
            df_res.p_sample = df_res.p_sample* df.p_sample.values
            
        df_res.p_sample /= df_res.p_sample.sum()
        
        return df_res

    return _inner

def BatchLimit_Factory(cls):
    '''
    Batch limited Dataloader that provides an upper limit for the number of mini batches per epoch
    max_batches: upper limit for minibatch count per epoch
        
    '''
    assert issubclass(cls, TfmdDL)
    
    class BatchLimitDL(cls):
        def __init__(self, dataset, max_batches=None, **kwargs):
            self.max_batches = max_batches
            # kwargs['n'] = max_batches*kwargs['bs'] n has to remain the full size, in order to create all indices if shuffled
            super().__init__(dataset=dataset, **kwargs)

        def __len__(self):
            batches = super().__len__() 
            if self.max_batches is not None: 
                batches = min(batches,self.max_batches)
            return batches

        def __iter__(self):
            if self.max_batches is None: 
                yield from super().__iter__()
            else:
                for idx,b in enumerate(super().__iter__()):
                    if idx >= self.max_batches: 
                        break
                    yield b
                    
        #shuffle function that is called in super().get_idxs, truncated for faster execution
        def shuffle_fn(self, idxs): return self.rng.sample(idxs, min(len(self)*self.bs,len(idxs)))
                    
        #get_idxs is truncated for the non-shuffling case, otherwise shuffle_fn is already truncated
        def get_idxs(self):
            return super().get_idxs()[:len(self)*self.bs]

    return BatchLimitDL

def NBatches_Factory(cls):
    '''
    Fixed batch count dataloader that samples exactly `n_batches` per epoch.
    Oversamples (with replacement) if fewer samples exist, undersamples if more exist.
    Fully composable with other factories like `WeightedDL_Factory`.
    
    n_batches: exact number of minibatches per epoch (None disables fixed batching)
    '''
    assert issubclass(cls, TfmdDL)
    
    class NBatchesDL(cls):
        def __init__(self, 
                     dataset, # dataset to sample from
                     n_batches=None, # target number of batches per epoch (None for original behavior)
                     **kwargs):
            self.n_batches_target = n_batches
            super().__init__(dataset=dataset, **kwargs)

        def __len__(self):
            "Returns `n_batches_target` if set, otherwise delegates to parent"
            if self.n_batches_target is None:
                return super().__len__()
            return self.n_batches_target

        def get_idxs(self):
            "Adjusts parent's indices to match `n_batches_target * bs` total samples"
            parent_idxs = super().get_idxs()
            
            if self.n_batches_target is None:
                return parent_idxs
            
            if self.n == 0 or self.n_batches_target <= 0:
                return []
            
            bs = self.bs if self.bs is not None else 1
            target_samples = self.n_batches_target * bs
            n_parent = len(parent_idxs)
            
            if n_parent == target_samples:
                return parent_idxs
            elif n_parent > target_samples:
                return parent_idxs[:target_samples]
            else:
                # Oversample preserving parent's sampling logic
                if self.shuffle:
                    return self.rng.choices(parent_idxs, k=target_samples)
                else:
                    full_cycles = target_samples // n_parent
                    remainder = target_samples % n_parent
                    return parent_idxs * full_cycles + parent_idxs[:remainder]

    return NBatchesDL

def get_inp_out_size(dls):
    '''returns input and output size of a timeseries databunch'''
    tup = dls.one_batch()
    inp = tup[0].shape[-1]
    out = tup[1].shape[-1]
    return inp,out

