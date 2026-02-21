
__all__ = ['PredictionCallback']

from ..data import *
from ..models import *
from ..learner import *
from fastai.basics import *

class PredictionCallback(Callback):
    'Concatenates the system output to the input data for autoregression, assumes 1-tuple as input'

    order= -56 #the callback has to be the first one executed, so everything else has the correct database

    def __init__(self,
                 t_offset=1 #the number of steps output is shifted in the past, shortens the sequence length by that number
                ):
        super().__init__()
        self.t_offset = t_offset

    def before_batch(self):
        #output has to be casted to the input tensor type
        x = self.x
        y = self.yb[0].as_subclass(type(x))

        if self.t_offset != 0:
            x = x[:,self.t_offset:,:]
            y = y[:,:-self.t_offset,:]

            #shorten the output by the same size
            self.learn.yb = tuple((y[:,self.t_offset:,:] for y in self.yb))

        #concatenate and reconvert to tuple
        self.learn.xb = (torch.cat((x,y), dim=-1),)
