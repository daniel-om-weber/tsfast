# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/01_models/02_cnn.ipynb.

# %% auto 0
__all__ = ['CausalConv1d', 'CConv1D', 'TCN_Block', 'TCN', 'SeperateTCN', 'CRNN', 'SeperateCRNN']

# %% ../../nbs/01_models/02_cnn.ipynb 2
from ..data import *
from .layers import *
from .rnn import *
from fastai.basics import *
from torch.nn.utils import weight_norm

# %% ../../nbs/01_models/02_cnn.ipynb 7
class CausalConv1d(torch.nn.Conv1d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 groups=1,
                 bias=True,
                stateful=False):

        super().__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias)
        self.__init_size = (kernel_size - 1) * dilation
        self.x_init = None
        self.stateful = stateful
        
    def forward(self, x):
        if self.x_init is not None and self.x_init.shape[0] != x.shape[0]: 
            self.x_init = None
        
        if self.x_init is None or not self.stateful:
            self.x_init = torch.zeros((x.shape[0],x.shape[1],self.__init_size),device=x.device)
            
        x = torch.cat([self.x_init,x],dim=-1)
        
        out = super().forward(x)
    
        if self.stateful: self.x_init =to_detach( x[...,-self.__init_size:], cpu=False, gather=False) 
            
        return out
    
    def reset_state(self):
        self.x_init = None

# %% ../../nbs/01_models/02_cnn.ipynb 8
@delegates(CausalConv1d, keep=True)
def CConv1D(input_size,output_size,kernel_size=2,activation = Mish,wn=True, bn = False, **kwargs):
    conv = CausalConv1d(input_size,output_size,kernel_size,**kwargs)
    if wn: conv = weight_norm(conv)
    act = activation() if activation is not None else None
    bn = nn.BatchNorm1d(input_size) if bn else None
    m = [m for m in [bn,conv,act] if m is not None]
    return nn.Sequential(*m)

# %% ../../nbs/01_models/02_cnn.ipynb 9
@delegates(CausalConv1d, keep=True)
class TCN_Block(nn.Module):
    def __init__(self,input_size,output_size,num_layers=1,
                 activation = Mish,wn=True, bn = False,stateful=False, **kwargs):
        super().__init__()
        
        layers=[]
        for _ in range(num_layers):
            conv = CausalConv1d(input_size,output_size,2,stateful=stateful,**kwargs)
            if wn: conv = weight_norm(conv)
            act = activation() if activation is not None else None
            bn = nn.BatchNorm1d(input_size) if bn else None
            layers += [m for m in [bn,conv,act] if m is not None]
            
        self.layers = nn.Sequential(*layers)
        
        self.residual = nn.Conv1d(input_size,output_size,kernel_size=1) if output_size!=input_size else None
        
    def forward(self, x):      
        out = self.layers(x)
        out = out + (x if self.residual is None else self.residual(x))  
        return out

# %% ../../nbs/01_models/02_cnn.ipynb 10
class TCN(nn.Module):
    def __init__(self,input_size,output_size,hl_depth=1,hl_width=10,act = Mish,bn=False,stateful=False):
        super().__init__()
        
        conv_layers = [TCN_Block(input_size if i==0 else hl_width,hl_width,
                                      dilation=2**(i),bn=bn,activation=act,stateful=stateful)
                                          for i in range(hl_depth)]
        self.conv_layers = nn.Sequential(*conv_layers)
        self.final = nn.Conv1d(hl_width,output_size,kernel_size=1)
        
    def forward(self, x):      
        x_in = x.transpose(1,2)
        out = self.conv_layers(x_in)
        out = self.final(out).transpose(1,2)
        return out

# %% ../../nbs/01_models/02_cnn.ipynb 12
class SeperateTCN(nn.Module):
    def __init__(self,input_list,output_size,hl_depth=1,hl_width=10,act = Mish,bn=False,stateful=False,final_layer=3):
        super().__init__()
        self.input_list =np.cumsum([0] + input_list)
        
        tcn_width = hl_width//len(input_list)
        layers = [ [TCN_Block(n if i==0 else tcn_width,tcn_width,
                                      dilation=2**(i),bn=bn,activation=act)
                                          for i in range(hl_depth)]
                        for n in input_list]
        self.layers = nn.ModuleList([nn.Sequential(*l) for l in layers])
        
        self.rec_field = (2**hl_depth)-1
        self.final = SeqLinear(tcn_width*len(input_list),output_size,hidden_size=hl_width,hidden_layer=final_layer)
        self.x_init = None
        self.stateful = stateful
        
    def forward(self, x):      
        if self.x_init is not None:
            if self.x_init.shape[0] != x.shape[0]: 
                self.x_init = None
            elif self.stateful: 
                x = torch.cat([self.x_init,x],dim=1)
        
        tcn_out = [layer(x[...,self.input_list[i]:self.input_list[i+1]].transpose(1,2)) 
                   for i,layer in enumerate(self.layers)]
        out = torch.cat(tcn_out,dim=1).transpose(1,2)
        
        out = self.final(out)
        
        if self.stateful:
            if self.x_init is not None: out = out[:,self.rec_field:]
            self.x_init = x[:,-self.rec_field:]
            
        return out
    
    def reset_state(self):
        self.x_init = None

# %% ../../nbs/01_models/02_cnn.ipynb 14
class CRNN(nn.Module):
    def __init__(self,input_size,output_size,num_ft=10,num_cnn_layers=4,num_rnn_layers=2,hs_cnn=10,hs_rnn=10,
         hidden_p=0, input_p=0, weight_p=0, rnn_type='gru',stateful=False):
        super().__init__()
        self.cnn = TCN(input_size,num_ft,num_cnn_layers,hs_cnn,act=nn.ReLU,stateful=stateful)
        self.rnn = SimpleRNN(num_ft,output_size,num_layers=num_rnn_layers,hidden_size=hs_rnn,
                       hidden_p=hidden_p, input_p=input_p, weight_p=weight_p, 
                        rnn_type=rnn_type,stateful=stateful)
        
    def forward(self, x):
        return self.rnn(self.cnn(x))
    

# %% ../../nbs/01_models/02_cnn.ipynb 17
class SeperateCRNN(nn.Module):
    def __init__(self,input_list,output_size,num_ft=10,num_cnn_layers=4,num_rnn_layers=2,hs_cnn=10,hs_rnn=10,
         hidden_p=0, input_p=0, weight_p=0, rnn_type='gru',stateful=False):
        super().__init__()
        self.cnn = SeperateTCN(input_list,num_ft,num_cnn_layers,hs_cnn,act=nn.ReLU,stateful=stateful,final_layer=0)
        self.rnn = SimpleRNN(num_ft,output_size,num_layers=num_rnn_layers,hidden_size=hs_rnn,
                       hidden_p=hidden_p, input_p=input_p, weight_p=weight_p, 
                        rnn_type=rnn_type,stateful=stateful)
        
    def forward(self, x):
        return self.rnn(self.cnn(x))