"""Reusable model layers, scalers, and wrappers for normalization and aggregation."""

__all__ = [
    "BatchNorm_1D_Stateful",
    "SeqLinear",
    "Scaler",
    "StandardScaler1D",
    "MinMaxScaler1D",
    "MaxAbsScaler1D",
    "AR_Model",
    "NormalizedModel",
    "unwrap_model",
    "SeqAggregation",
]

from ..data import *
from fastai.basics import *

from fastai.callback.progress import *  # import activate learning progress bar
from torch.nn import Parameter


class BatchNorm_1D_Stateful(nn.Module):
    """Batchnorm for stateful models. Stores batch statistics for for every timestep seperately to mitigate transient effects."""

    __constants__ = [
        "track_running_stats",
        "momentum",
        "eps",
        "weight",
        "bias",
        "running_mean",
        "running_var",
        "num_batches_tracked",
    ]

    def __init__(
        self,
        hidden_size,
        seq_len=None,
        stateful=False,
        batch_first=True,
        eps=1e-7,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
    ):  # num_features
        super().__init__()
        channel_d = hidden_size
        self.seq_len = seq_len
        self.stateful = stateful
        self.batch_first = batch_first
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        self.axes = (1,)
        if self.affine:
            self.weight = Parameter(torch.Tensor(channel_d))
            self.bias = Parameter(torch.Tensor(channel_d))
            self.register_parameter("weight", self.weight)
            self.register_parameter("bias", self.bias)
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        if self.track_running_stats:
            if seq_len is not None:
                self.register_buffer("running_mean", torch.zeros(seq_len, channel_d))
                self.register_buffer("running_var", torch.ones(seq_len, channel_d))
            self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter("running_mean", None)
            self.register_parameter("running_var", None)
            self.register_parameter("num_batches_tracked", None)
        self.reset_parameters()
        self.reset_state()

    def reset_state(self):
        self.seq_idx = 0

    def reset_parameters(self):
        if self.track_running_stats and self.seq_len is not None:
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.num_batches_tracked.zero_()
        if self.affine:
            self.weight.data.fill_(1.0)
            self.bias.data.fill_(0.0)

    def forward(self, input, BN_start=None):
        if input.dim() != 3:
            raise ValueError("expected 3D input (got {}D input)".format(input.dim()))
        if self.batch_first:
            input = input.transpose(0, 1)

        input_t, n_batch, hidden_size = input.size()

        if self.track_running_stats and self.seq_len is None:
            self.seq_len = input_t
            self.register_buffer("running_mean", torch.zeros((input_t, hidden_size), device=input.device))
            self.register_buffer("running_var", torch.ones((input_t, hidden_size), device=input.device))

        if BN_start is None:
            if self.stateful:
                BN_start = self.seq_idx
            else:
                BN_start = 0

        exponential_average_factor = 0.0
        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        BN_stop = BN_start + input_t
        self.seq_idx = BN_stop  # new starting point for next forward call

        if self.training:
            mean = input.mean(1)
            var = input.var(1, unbiased=False)  # use biased var in train

            if self.seq_len - BN_start > 0:  # frame has to be in statistics window for updates
                with torch.no_grad():
                    self.running_mean[BN_start:BN_stop] = (
                        exponential_average_factor * mean[: self.seq_len - BN_start]
                        + (1 - exponential_average_factor) * self.running_mean[BN_start:BN_stop]
                    )
                    self.running_var[BN_start:BN_stop] = (
                        exponential_average_factor * var[: self.seq_len - BN_start] * n_batch / (n_batch - 1)
                        + (1 - exponential_average_factor) * self.running_var[BN_start:BN_stop]
                    )  # update running_var with unbiased var
        else:
            mean = self.running_mean[BN_start:BN_stop]
            var = self.running_var[BN_start:BN_stop]

            # if elements outside of the statistics are requested, append the last element repeatedly
            #             import pdb;pdb.set_trace()
            if BN_stop >= self.seq_len:
                cat_len = input_t - max(self.seq_len - BN_start, 0)  # min(BN_stop-self.seq_len,self.seq_len)
                mean = torch.cat((mean, self.running_mean[-1:].repeat(cat_len, 1)))
                var = torch.cat((var, self.running_var[-1:].repeat(cat_len, 1)))

        output = (input - mean[:, None, :]) / (torch.sqrt(var[:, None, :] + self.eps))
        if self.affine:
            output = output * self.weight[None, None, :] + self.bias[None, None, :]  # [None, :, None, None]

        if self.batch_first:
            output = output.transpose(0, 1)

        return output


class SeqLinear(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=100, hidden_layer=1, act=Mish, batch_first=True):
        super().__init__()
        self.batch_first = batch_first

        def conv_act(inp, out):
            return nn.Sequential(nn.Conv1d(inp, out, 1), act())

        if hidden_layer < 1:
            self.lin = nn.Conv1d(input_size, output_size, 1)
        else:
            self.lin = nn.Sequential(
                conv_act(input_size, hidden_size),
                *[conv_act(hidden_size, hidden_size) for _ in range(hidden_layer - 1)],
                nn.Conv1d(hidden_size, output_size, 1),
            )

    def forward(self, x):
        if not self.batch_first:
            x = x.transpose(0, 1)
        out = self.lin(x.transpose(1, 2)).transpose(1, 2)

        if not self.batch_first:
            out = out.transpose(0, 1)
        return out


class Scaler(nn.Module):
    "Base class for feature scaling on [batch, seq, features] tensors."

    def normalize(self, x):
        raise NotImplementedError

    def denormalize(self, x):
        raise NotImplementedError

    def unnormalize(self, x):
        return self.denormalize(x)

    @classmethod
    def from_stats(cls, stats):
        "Create a Scaler from a NormPair. Override in subclasses."
        raise NotImplementedError(f"{cls.__name__} must implement from_stats")


def _ensure_tensor(arr):
    "Convert numpy array or tensor to shape [1,1,features] float tensor."
    if isinstance(arr, np.ndarray):
        return torch.from_numpy(arr[None, None, :]).float()
    if isinstance(arr, torch.Tensor):
        if arr.ndim == 1:
            return arr[None, None, :].float()
        return arr.float()
    raise TypeError(f"Expected ndarray or Tensor, got {type(arr)}")


class StandardScaler1D(Scaler):
    "Normalize by (x - mean) / std."

    _epsilon = 1e-16

    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", _ensure_tensor(mean))
        self.register_buffer("std", _ensure_tensor(std) + self._epsilon)

    def normalize(self, x):
        return (x - self.mean) / self.std

    def denormalize(self, x):
        return x * self.std + self.mean

    @classmethod
    def from_stats(cls, stats):
        return cls(stats.mean, stats.std)


class MinMaxScaler1D(Scaler):
    "Normalize by (x - min) / (max - min) to [0, 1]."

    _epsilon = 1e-16

    def __init__(self, min_val, max_val):
        super().__init__()
        self.register_buffer("min_val", _ensure_tensor(min_val))
        self.register_buffer("range_val", _ensure_tensor(max_val) - _ensure_tensor(min_val) + self._epsilon)

    def normalize(self, x):
        return (x - self.min_val) / self.range_val

    def denormalize(self, x):
        return x * self.range_val + self.min_val

    @classmethod
    def from_stats(cls, stats):
        return cls(stats.min, stats.max)


class MaxAbsScaler1D(Scaler):
    "Normalize by x / max(|min|, |max|)."

    _epsilon = 1e-16

    def __init__(self, min_val, max_val):
        super().__init__()
        self.register_buffer(
            "max_abs", torch.max(torch.abs(_ensure_tensor(min_val)), torch.abs(_ensure_tensor(max_val))) + self._epsilon
        )

    def normalize(self, x):
        return x / self.max_abs

    def denormalize(self, x):
        return x * self.max_abs

    @classmethod
    def from_stats(cls, stats):
        return cls(stats.min, stats.max)


class AR_Model(nn.Module):
    """
    Autoregressive model container which work autoregressively if the sequence y is not provided, otherwise it works as a normal model.
    This way it can be trained either with teacher forcing or with autoregression.
    Normalization should be handled externally via NormalizedModel wrapping.
    """

    def __init__(self, model, ar=True, stateful=False, model_has_state=False, return_state=False, out_sz=None):
        super().__init__()
        self.model = model
        self.ar = ar
        self.stateful = stateful
        self.model_has_state = model_has_state
        self.return_state = return_state
        self.out_sz = out_sz
        if return_state and not model_has_state:
            raise ValueError("return_state=True requires model_has_state=True")
        self.y_init = None

    def forward(self, inp, h_init=None, ar=None):
        if ar is None:
            ar = self.ar

        if ar:  # autoregressive mode
            y_e = []

            y_next = (
                self.y_init if self.y_init is not None else torch.zeros(inp.shape[0], 1, self.out_sz).to(inp.device)
            )

            # two loops in the if clause to avoid the if inside the loop
            if self.model_has_state:
                h0 = h_init
                for u_in in inp.split(1, dim=1):
                    x = torch.cat((u_in, y_next), dim=2)
                    y_next, h0 = self.model(x, h0)
                    y_e.append(y_next)
            else:
                for u_in in inp.split(1, dim=1):
                    x = torch.cat((u_in, y_next), dim=2)
                    y_next = self.model(x)
                    y_e.append(y_next)

            y_e = torch.cat(y_e, dim=1)

        else:  # teacherforcing mode
            if self.model_has_state:
                y_e, h0 = self.model(inp, h_init)
            else:
                y_e = self.model(inp)

        if self.stateful:
            self.y_init = to_detach(y_e[:, -1:], cpu=False, gather=False)

        return y_e if not self.return_state else (y_e, h0)

    def reset_state(self):
        self.y_init = None


class NormalizedModel(nn.Module):
    "Wraps a model with input normalization and optional output denormalization."

    def __init__(self, model, input_norm: Scaler, output_norm: Scaler | None = None):
        super().__init__()
        self.model = model
        self.input_norm = input_norm
        self.output_norm = output_norm

    @classmethod
    def from_stats(cls, model, input_stats, output_stats=None, scaler_cls=None):
        "Create from NormPair stats with the given Scaler class."
        if scaler_cls is None:
            scaler_cls = StandardScaler1D
        input_norm = scaler_cls.from_stats(input_stats)
        output_norm = scaler_cls.from_stats(output_stats) if output_stats is not None else None
        return cls(model, input_norm, output_norm)

    @classmethod
    def from_dls(cls, model, dls, prediction=False, scaler_cls=None):
        "Create from DataLoaders norm_stats."
        from ..datasets.core import extract_mean_std_from_dls

        norm_u, norm_x, norm_y = extract_mean_std_from_dls(dls)
        if prediction:
            parts = [norm_u] + ([norm_x] if norm_x else []) + [norm_y]
            input_stats = sum(parts[1:], parts[0])
        else:
            input_stats = norm_u
        return cls.from_stats(model, input_stats, norm_y, scaler_cls=scaler_cls)

    def forward(self, xb, **kwargs):
        xb = self.input_norm.normalize(xb)
        out = self.model(xb, **kwargs)
        if self.output_norm is not None:
            out = self.output_norm.denormalize(out)
        return out


def _unwrap_ddp(model):
    "Unwrap DistributedDataParallel/DataParallel wrappers."
    while hasattr(model, "module"):
        model = model.module
    return model


def unwrap_model(model):
    "Get the inner model, unwrapping DDP/DP and NormalizedModel if present."
    model = _unwrap_ddp(model)
    return model.model if isinstance(model, NormalizedModel) else model


class SeqAggregation(nn.Module):
    """Aggregation layer that reduces the sequence dimension.

    Args:
        func: aggregation function taking (tensor, dim) and returning reduced tensor
        dim: sequence dimension to aggregate over
    """

    def __init__(
        self,
        func: callable = lambda x, dim: x.select(dim, -1),
        dim: int = 1,
    ):
        super().__init__()
        self.func = func
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        "Apply the aggregation function to the input tensor."
        return self.func(x, dim=self.dim)
