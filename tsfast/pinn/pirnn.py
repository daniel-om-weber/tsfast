
__all__ = ['PIRNN', 'AuxiliaryOutputLoss', 'PIRNNLearner']

import torch
import torch.nn as nn
from ..prediction.fransys import Diag_RNN
from ..models.rnn import RNN
from ..models.layers import SeqLinear, Scaler, NormalizedModel, _resolve_norm
from ..learner.callbacks import CB_TruncateSequence
from ..learner.losses import SkipNLoss
from fastai.basics import *
from fastcore.basics import store_attr
from functools import partial


class PIRNN(nn.Module):
    '''Physics-Informed RNN with dual encoders: Sequence and State'''
    
    def __init__(self,
                 n_u:int, # Number of inputs
                 n_y:int, # Number of outputs (total: supervised + auxiliary)
                 init_sz:int, # Initialization sequence length
                 n_y_supervised:int = None, # Number of supervised outputs (in dataset)
                 n_x:int = 0, # Number of extra states
                 hidden_size:int = 100, # Hidden state size
                 rnn_layer:int = 1, # Number of RNN layers
                 state_encoder_hidden:int = 64, # Hidden size for state encoder MLP
                 linear_layer:int = 1, # Linear layers in diagnosis RNN
                 final_layer:int = 0, # Final layer complexity
                 init_diag_only:bool = False, # Limit diagnosis to init_sz
                 default_encoder_mode:str = 'sequence', # Default encoder mode
                 **kwargs
                ):
        '''Initialize PIRNN with both sequence and state encoders'''
        super().__init__()
        n_y_supervised = n_y_supervised if n_y_supervised is not None else n_y
        store_attr('n_u,n_y,n_x,n_y_supervised,init_sz,init_diag_only,hidden_size,rnn_layer,default_encoder_mode')
        
        # Instantiate FranSys components - diagnosis RNN uses supervised outputs only
        self.rnn_diagnosis = Diag_RNN(
            n_u+n_x+n_y_supervised, hidden_size, 
            hidden_size=hidden_size,
            output_layer=rnn_layer,
            rnn_layer=rnn_layer,
            linear_layer=linear_layer,
            **kwargs
        )
        
        rnn_kwargs = dict(hidden_size=hidden_size, num_layers=rnn_layer, ret_full_hidden=True)
        rnn_kwargs = dict(rnn_kwargs, **kwargs)
        self.rnn_prognosis = RNN(n_u, **rnn_kwargs)
        
        # Final layer outputs all channels (supervised + auxiliary)
        self.final = SeqLinear(hidden_size, n_y, hidden_layer=final_layer)
        
        # State encoder: physical state -> hidden state (uses supervised outputs)
        self.state_encoder = nn.Sequential(
            nn.Linear(n_y_supervised, state_encoder_hidden),
            nn.ReLU(),
            nn.Linear(state_encoder_hidden, hidden_size * rnn_layer)
        )
    
    def forward(
        self,
        x:torch.Tensor, # Input tensor
        init_state:list = None, # Initial hidden state (optional)
        encoder_mode:str = 'default' # 'none', 'sequence', or 'state'
    ) -> torch.Tensor: # Output predictions
        '''Forward pass with encoder mode auto-detection or explicit selection'''

        u = x[:,:,:self.n_u]
        # Use n_y_supervised for initialization sequence (only supervised outputs in data)
        x_init = x[:,:self.init_sz,:self.n_u+self.n_x+self.n_y_supervised]
        if encoder_mode == 'default':
            encoder_mode = self.default_encoder_mode
        
        # Detect encoder mode based on input shape
        if encoder_mode == 'none':
            return self._forward_predictor(u, init_state)
        elif encoder_mode == 'sequence':
            return self._forward_sequence_encoder(u[:,self.init_sz:], x_init, init_state)
        elif encoder_mode == 'state':
            return self._forward_state_encoder(u[:,self.init_sz:], x_init, init_state)
        else:
            raise ValueError(f"encoder_mode must be 'none', 'sequence', or 'state', got {encoder_mode}")
    
    def _forward_sequence_encoder(
        self,
        u:torch.Tensor, # Full input [batch, seq, n_u+n_x+n_y]
        x_init:torch.Tensor, # Initial state [batch, seq, n_x+n_y]
        init_state:list = None # Initial hidden state (optional)
    ) -> torch.Tensor: # Output predictions
        '''Forward using sequence encoder (diagnosis RNN)'''
        out_init,_ = self.rnn_diagnosis(x_init)
        if init_state is None:
            init_state = self.rnn_diagnosis.output_to_hidden(out_init, -1)
        out_prog,self.new_hidden = self.rnn_prognosis(u, init_state)
        out_prog = torch.cat([out_init, out_prog], 2) # [n_layers, batch, seq ,n_y]

        result = self.final(out_prog[-1])
        return result
    
    def _forward_state_encoder(
        self,
        u:torch.Tensor, # Prognosis input [batch, seq, n_u]
        x_init:torch.Tensor, # Init window input [batch, init_sz, n_x + n_y_supervised]
        init_state, 
    ) -> torch.Tensor: # Output predictions
        '''Forward using state encoder (MLP), zero-pads init window'''
        if init_state is None: # If init_state is not provided, use last initialization step
            init_state = x_init[:,-1,-self.n_y_supervised:]
        init_state = self.encode_single_state(init_state)
        pred = self._forward_predictor(u, init_state)
        return F.pad(pred, (0, 0, self.init_sz, 0)) # Zero-pad init window


    def _forward_predictor(
        self,
        u:torch.Tensor, # Input tensor
        init_state:list # Initial hidden state
    ) -> torch.Tensor: # Output predictions
        '''Forward using predictor RNN'''
        out_prog,_ = self.rnn_prognosis(u, init_state)
        return self.final(out_prog[-1])
    
    def encode_single_state(
        self,
        physical_state:torch.Tensor # Physical state [batch, n_y_supervised]
    ) -> list: # Hidden state compatible with RNN [rnn_layer, batch, hidden_size]
        '''Convert single physical state to RNN-compatible hidden state'''
        batch_size = physical_state.shape[0]
        
        # Encode: [batch, n_y_supervised] -> [batch, hidden_size * rnn_layer]
        h_flat = self.state_encoder(physical_state)
        
        # Reshape to RNN format: [rnn_layer, batch, hidden_size]
        h = h_flat.view(batch_size, self.rnn_layer, self.hidden_size)
        h = h.transpose(0, 1).contiguous()  # [rnn_layer, batch, hidden_size]
        
        # Convert to list format expected by Diag_RNN.output_to_hidden
        return [h[i:i+1] for i in range(self.rnn_layer)]

class AuxiliaryOutputLoss:
    '''Wrapper that applies loss only to supervised outputs, ignoring auxiliary outputs'''
    def __init__(
        self,
        loss_func, # Loss function to wrap
        n_supervised:int, # Number of supervised output channels
    ):
        '''Wrap loss function to only compute on supervised outputs'''
        store_attr()
    
    def __call__(
        self,
        pred:torch.Tensor, # Predictions [batch, seq, n_y_total]
        targ:torch.Tensor, # Targets [batch, seq, n_y_supervised]
    ) -> torch.Tensor: # Loss value
        '''Apply loss only to first n_supervised channels of predictions'''
        return self.loss_func(pred[..., :self.n_supervised], targ)

@delegates(PIRNN, keep=True)
def PIRNNLearner(
    dls, # DataLoaders
    init_sz:int, # Initialization sequence length
    n_aux_outputs:int = 0, # Number of auxiliary outputs (not in dataset)
    attach_output:bool = False, # Whether to attach output to input
    loss_func = nn.L1Loss(), # Loss function
    metrics = None, # Metrics
    opt_func = Adam, # Optimizer
    lr:float = 3e-3, # Learning rate
    cbs = None, # Additional callbacks
    input_norm = 'standard', # Input normalization method
    output_norm = False, # Output denormalization method
    **kwargs # Additional arguments for PIRNN
):
    '''Create PIRNN learner with appropriate configuration'''
    from tsfast.prediction.core import PredictionCallback
    from tsfast.learner.losses import fun_rmse

    cbs = [] if cbs is None else list(cbs)
    metrics = [fun_rmse] if metrics is None else list(metrics) if is_iter(metrics) else [metrics]

    _batch = dls.one_batch()
    inp = _batch[0].shape[-1]
    out = _batch[1].shape[-1]  # Supervised outputs from dataset
    n_y_total = out + n_aux_outputs  # Total outputs (supervised + auxiliary)

    norm_u, norm_x, norm_y = dls.norm_stats

    if attach_output:
        model = PIRNN(inp, n_y_total, init_sz, n_y_supervised=out, **kwargs)

        # Add PredictionCallback if not present
        if not any(isinstance(cb, PredictionCallback) for cb in cbs):
            cbs.append(PredictionCallback(0))

        # Input will be [u, y] after PredictionCallback concatenation
        combined_input_stats = norm_u + norm_y
    else:
        model = PIRNN(inp-out, n_y_total, init_sz, n_y_supervised=out, **kwargs)

        # Input is [u, x?, y] from prediction-mode dls
        parts = [norm_u] + ([norm_x] if norm_x else []) + [norm_y]
        combined_input_stats = sum(parts[1:], parts[0])

    # Wrap model with input normalization and optional output denormalization
    in_method = _resolve_norm(input_norm)
    out_method = _resolve_norm(output_norm)
    if in_method:
        in_scaler = Scaler.from_stats(combined_input_stats, in_method)
        out_scaler = Scaler.from_stats(norm_y, out_method) if out_method else None
        model = NormalizedModel(model, in_scaler, out_scaler)

    # For long sequences, add truncation callback
    seq_len = _batch[0].shape[1]
    LENGTH_THRESHOLD = 300
    if seq_len > init_sz + LENGTH_THRESHOLD:
        if not any(isinstance(cb, CB_TruncateSequence) for cb in cbs):
            INITIAL_SEQ_LEN = 100
            cbs.append(CB_TruncateSequence(init_sz + INITIAL_SEQ_LEN))

    # Wrap loss and metrics to only use supervised outputs when auxiliary outputs present
    if n_aux_outputs > 0:
        loss_func = AuxiliaryOutputLoss(loss_func, out)
        metrics = [AuxiliaryOutputLoss(m, out) for m in metrics]

    # Skip initial timesteps in loss/metrics
    skip = partial(SkipNLoss, n_skip=init_sz)
    metrics = [skip(f) for f in metrics]
    loss_func = skip(loss_func)

    lrn = Learner(dls, model, loss_func=loss_func, metrics=metrics, cbs=cbs, opt_func=opt_func, lr=lr)
    return lrn

