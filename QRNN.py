from fastai.basics import *
from fastai.text.models.qrnn import *
from fastai.text.models.awd_lstm import *


from fastai.text.models.qrnn import dispatch_cuda
from fastai.text.models.qrnn import ForgetMultGPU
from fastai.text.models.qrnn import BwdForgetMultGPU
from fastai.text.models.qrnn import forget_mult_CPU

class QRNNLayer(Module):
    "Apply a single layer Quasi-Recurrent Neural Network (QRNN) to an input sequence."

    def __init__(self, input_size:int, hidden_size:int=None, save_prev_x:bool=False, zoneout:float=0, window:int=1, 
                 output_gate:bool=True, batch_first:bool=True, backward:bool=False):
        super().__init__()
        assert window in [1, 2], "This QRNN implementation currently only handles convolutional window of size 1 or size 2"
        self.save_prev_x,self.zoneout,self.window = save_prev_x,zoneout,window
        self.output_gate,self.batch_first,self.backward = output_gate,batch_first,backward
        hidden_size = ifnone(hidden_size, input_size)
        #One large matmul with concat is faster than N small matmuls and no concat
        mult = (3 if output_gate else 2)
        self.linear = nn.Linear(window * input_size, mult * hidden_size)
        self.prevX = None

    def reset(self):
        # If you are saving the previous value of x, you should call this when starting with a new state
        self.prevX = None
        
    def forward(self, inp, hid=None):
        y = self.linear(self._get_source(inp))
        if self.output_gate: z_gate,f_gate,o_gate = y.chunk(3, dim=2)
        else:                z_gate,f_gate        = y.chunk(2, dim=2)
        z_gate.tanh_()
        f_gate.sigmoid_()
        if self.zoneout and self.training:
            mask = dropout_mask(f_gate, f_gate.size(), self.zoneout).requires_grad_(False)
            f_gate = f_gate * mask
        z_gate,f_gate = z_gate.contiguous(),f_gate.contiguous()
        if self.backward: forget_mult = dispatch_cuda(BwdForgetMultGPU, partial(forget_mult_CPU, backward=True), inp)
        else:             forget_mult = dispatch_cuda(ForgetMultGPU, forget_mult_CPU, inp)
        c_gate = forget_mult(z_gate, f_gate, hid, self.batch_first)
        output = torch.sigmoid(o_gate) * c_gate if self.output_gate else c_gate
        if self.window > 1 and self.save_prev_x: 
            if self.backward: self.prevX = (inp[:, :1] if self.batch_first else inp[:1]).detach()
            else:             self.prevX = (inp[:, -1:] if self.batch_first else inp[-1:]).detach()
        idx = 0 if self.backward else -1
        return output, (c_gate[:, idx] if self.batch_first else c_gate[idx]),c_gate

    def _get_source(self, inp):
        if self.window == 1: return inp
        dim = (1 if self.batch_first else 0)
        inp_shift = [torch.zeros_like(inp[:,:1] if self.batch_first else inp[:1]) if self.prevX is None else self.prevX]
        if self.backward: inp_shift.insert(0,inp[:,1:] if self.batch_first else inp[1:])
        else:             inp_shift.append(inp[:,:-1] if self.batch_first else inp[:-1])
        inp_shift = torch.cat(inp_shift, dim)
        return torch.cat([inp, inp_shift], 2)
    
class QRNN(Module):
    "Apply a multiple layer Quasi-Recurrent Neural Network (QRNN) to an input sequence."

    def __init__(self, input_size:int, hidden_size:int, n_layers:int=1, bias:bool=True, batch_first:bool=True,
                 dropout:float=0, bidirectional:bool=False, save_prev_x:bool=False, zoneout:float=0, window:int=None, 
                 output_gate:bool=True):
        assert not (save_prev_x and bidirectional), "Can't save the previous X with bidirectional."
        assert bias == True, 'Removing underlying bias is not yet supported'
        super().__init__()
        kwargs = dict(batch_first=batch_first, zoneout=zoneout, output_gate=output_gate)
        self.layers = nn.ModuleList([QRNNLayer(input_size if l == 0 else hidden_size, hidden_size, save_prev_x=save_prev_x, 
                                               window=((2 if l ==0 else 1) if window is None else window), **kwargs) 
                                     for l in range(n_layers)])
        if bidirectional:
            self.layers_bwd = nn.ModuleList([QRNNLayer(input_size if l == 0 else hidden_size, hidden_size, 
                                                       backward=True, window=((2 if l ==0 else 1) if window is None else window), 
                                                       **kwargs) for l in range(n_layers)])
        self.n_layers,self.batch_first,self.dropout,self.bidirectional = n_layers,batch_first,dropout,bidirectional
        
    def reset(self):
        "If your convolutional window is greater than 1 and you save previous xs, you must reset at the beginning of each new sequence."
        for layer in self.layers:     layer.reset()
        if self.bidirectional:
            for layer in self.layers_bwd: layer.reset()    

    def forward(self, inp, hid=None):
        new_hid = []
        new_cgates = []
        if self.bidirectional: inp_bwd = inp.clone()
        for i, layer in enumerate(self.layers):
            inp, h,c_gate = layer(inp, None if hid is None else hid[2*i if self.bidirectional else i])
            new_hid.append(h)
            new_cgates.append(c_gate)
            
            if self.bidirectional:
                inp_bwd, h_bwd = self.layers_bwd[i](inp_bwd, None if hid is None else hid[2*i+1])
                new_hid.append(h_bwd)
            if self.dropout != 0 and i < len(self.layers) - 1:
                for o in ([inp, inp_bwd] if self.bidirectional else [inp]):
                    o = F.dropout(o, p=self.dropout, training=self.training, inplace=False)
        if self.bidirectional: inp = torch.cat([inp, inp_bwd], dim=2)
        return inp, torch.stack(new_hid, 0),torch.stack(new_cgates, 0)