
__all__ = ['SeqSlice', 'SeqNoiseInjection', 'SeqNoiseInjection_Varying', 'SeqNoiseInjection_Grouped', 'SeqBiasInjection',
           'encodes', 'decodes']

from fastai.basics import *
from .core import TensorSequencesInput
from fastai.vision.augment import RandTransform

class SeqSlice(Transform):
    '''Take a slice from an array-like object. Useful for e.g. shifting input and output'''
    def __init__(self, l_slc=None,r_slc=None):
        self.l_slc,self.r_slc = l_slc,r_slc
        
    def encodes(self, o): return o[self.l_slc:self.r_slc]

class SeqNoiseInjection(RandTransform):
    split_idx=0 #apply only to training data, if None it will be applied to all data
    '''Adds normal distributed noise to the tensor sequence with seperate mean and std for every signal'''
    def __init__(self, std=1e-1,mean=0.,p=1.0):
        super().__init__(p=p)
        self.std = tensor(std).type(torch.float)
        self.mean = tensor(mean).type(torch.float)
        
    def encodes(self, o:TensorSequencesInput): 
        if o.device != self.mean.device:
            self.std = self.std.to(o.device)
            self.mean = self.mean.to(o.device)
        #expand creates a view on a tensor and is therefore very fast compared to copy
        return o+torch.normal(mean=self.mean.expand_as(o), 
                              std=self.std.expand_as(o))

class SeqNoiseInjection_Varying(RandTransform):
    split_idx=0
    '''Adds normal distributed noise to the tensor sequence with a normal distributed standard deviation for every application'''
    def __init__(self, std_std=0.1,p=1.0):
        super().__init__(p=p)
        self.std_std = tensor(std_std).type(torch.float)
        
    def encodes(self, o:TensorSequencesInput): 
        if o.device != self.std_std.device:
            self.std_std = self.std_std.to(o.device)
            
        #expand creates a view on a tensor and is therefore very fast compared to copy
        std = torch.normal(mean=0,std=self.std_std).abs()
        return o+torch.normal(mean=0,std=std.expand_as(o))

class SeqNoiseInjection_Grouped(RandTransform):
    split_idx=0
    '''Adds normal distributed noise to the tensor sequence with a normal distributed standard deviation for every application, every group gert'''
    def __init__(self, std_std,std_idx,p=1.0):
        super().__init__(p=p)
        self.std_std = tensor(std_std).type(torch.float)
        self.std_idx = tensor(std_idx).type(torch.long)
        
    def encodes(self, o:TensorSequencesInput): 
        if o.device != self.std_std.device:
            self.std_std = self.std_std.to(o.device)
            
        #expand creates a view on a tensor and is therefore very fast compared to copy
        std = torch.normal(mean=0,std=self.std_std).abs()[self.std_idx]
        return o+torch.normal(mean=0,std=std.expand_as(o))

class SeqBiasInjection(RandTransform):
    split_idx=0
    '''Adds a normal distributed offset to the tensor sequence with seperate mean and std for every signal'''
    def __init__(self, std=1e-1,mean=0.,p=1.0):
        super().__init__(p=p)
        self.std = tensor(std).type(torch.float)
        self.mean = tensor(mean).type(torch.float)
        
    def encodes(self, o:TensorSequencesInput): 
        if o.device != self.mean.device:
            self.std = self.std.to(o.device)
            self.mean = self.mean.to(o.device)
        
        #expand creates a view on a tensor and is therefore very fast compared to copy
        mean=self.mean.repeat((o.shape[0],1,1)).expand((o.shape[0],1,o.shape[2]))
        std= self.std.repeat((o.shape[0],1,1)).expand((o.shape[0],1,o.shape[2]))
        n = torch.normal(mean=mean, std=std).expand_as(o)
        return o+n

@Normalize
def encodes(self, x:TensorSequencesInput): 
    if x.device != self.mean.device:
        self.mean = self.mean.to(x.device)
        self.std = self.std.to(x.device)
    return (x-self.mean) / self.std

@Normalize
def decodes(self, x:TensorSequencesInput):
    if x.device != self.mean.device:
        self.mean = self.mean.to(x.device)
        self.std = self.std.to(x.device)
    return (x*self.std + self.mean)
