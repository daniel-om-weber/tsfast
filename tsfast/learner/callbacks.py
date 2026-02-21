"""Training callbacks for gradient control, sequence manipulation, and regularization."""

__all__ = [
    "GradientClipping",
    "GradientNormPrint",
    "GradientBatchFiltering",
    "WeightClipping",
    "SkipFirstNCallback",
    "SkipNaNCallback",
    "CancelNaNCallback",
    "VarySeqLen",
    "sched_lin_p",
    "sched_ramp",
    "CB_TruncateSequence",
    "CB_AddLoss",
    "BatchLossFilter",
    "TimeSeriesRegularizer",
    "ARInitCB",
    "plot_grad_flow",
    "CB_PlotGradient",
]

from ..data import *
from fastai.basics import *


class GradientClipping(Callback):
    """Clips the gradient of every minibatch at a given value.

    Args:
        clip_val: maximum gradient norm threshold
    """

    def __init__(self, clip_val=10):
        self.clip_val = clip_val

    def after_backward(self):
        nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_val)


class GradientNormPrint(Callback):
    """Prints the norm of the gradient of every minibatch."""

    # def __init__(self, clip_val=10): self.clip_val = clip_val

    def before_step(self):
        grads = [param.grad.detach().flatten() for param in self.model.parameters() if param.grad is not None]
        norm = torch.cat(grads).norm()
        print(f"Gradient norm: {norm:.2f}")


class GradientBatchFiltering(Callback):
    """Skips batches with a gradient norm larger than a given threshold.

    Args:
        filter_val: maximum gradient norm before the batch is skipped
    """

    def __init__(self, filter_val=10):
        self.filter_val = filter_val

    def before_step(self):
        grads = [param.grad.detach().flatten() for param in self.model.parameters() if param.grad is not None]
        norm = torch.cat(grads).norm()
        if norm > self.filter_val:
            self.opt.zero_grad()
            # print(f'Gradient norm: {norm:.2f} filtered')
            raise CancelBatchException()
        # print(f'Gradient norm: {norm:.2f}')


class WeightClipping(Callback):
    """Clips the weights of a given module after every iteration.

    Args:
        module: the module whose weights are clipped
        clip_limit: symmetric clamp boundary for the weights
    """

    def __init__(self, module, clip_limit=1):
        self.module = module
        self.clip_limit = clip_limit

    def after_batch(self):
        #         import pdb; pdb.set_trace()
        for p in self.module.parameters():
            p.data.clamp_(-self.clip_limit, self.clip_limit)


class SkipFirstNCallback(Callback):
    """Skips first n time steps from prediction and target during training.

    Args:
        n_skip: number of initial time steps to discard
    """

    def __init__(self, n_skip=0):
        self.n_skip = n_skip

    def after_pred(self):
        if self.training:
            with torch.no_grad():
                dl = self.learn.dls.train
                if (hasattr(dl, "rnn_reset") and dl.rnn_reset) or not hasattr(
                    dl, "rnn_reset"
                ):  # if tbptt is used, only skip loss in the first minibatch
                    self.learn.pred = self.pred[:, self.n_skip :]
                    #         import pdb; pdb.set_trace()
                    if isinstance(self.yb, tuple):
                        self.learn.yb = tuple([y[:, self.n_skip :] for y in self.yb])
                    else:
                        self.learn.yb = self.yb[:, self.n_skip :]


class SkipNaNCallback(Callback):
    """Skips minibatches with a NaN loss."""

    def after_loss(self):
        #         import pdb;pdb.set_trace()
        if torch.isnan(self.learn.loss):
            self.opt.zero_grad()
            raise CancelBatchException()


class CancelNaNCallback(Callback):
    """Cancels training when a NaN loss is encountered."""

    def after_loss(self):
        if torch.isnan(self.learn.loss):
            raise CancelTrainException()


class VarySeqLen(Callback):
    """Randomly varies sequence length of every minibatch during training.

    Args:
        min_len: minimum sequence length to keep
    """

    def __init__(self, min_len=50):
        self.min_len = min_len

    def before_batch(self):
        if self.training:
            with torch.no_grad():
                seq_len_x = self.xb[0].shape[1]
                ly = self.yb[0].shape[1]
                lim = random.randint(self.min_len, ly)
                if ly < seq_len_x:
                    self.learn.xb = tuple([x[:, : -(ly - lim)] for x in self.xb])
                else:
                    self.learn.xb = tuple([x[:, :lim] for x in self.xb])

                self.learn.yb = tuple([y[:, :lim] for y in self.yb])


def sched_lin_p(start, end, pos, p=0.75):
    """Linear schedule that reaches the end value at position p.

    Args:
        start: value at position 0
        end: value at position p and beyond
        pos: current position in [0, 1]
        p: position at which the end value is reached
    """
    return end if pos >= p else start + pos / p * (end - start)


def sched_ramp(start, end, pos, p_left=0.2, p_right=0.6):
    """Ramp schedule that linearly transitions between two plateau regions.

    Args:
        start: value before p_left
        end: value after p_right
        pos: current position in [0, 1]
        p_left: position where the ramp begins
        p_right: position where the ramp ends
    """
    if pos >= p_right:
        return end
    elif pos <= p_left:
        return start
    else:
        return start + (end - start) * (pos - p_left) / (p_right - p_left)


from fastai.callback.all import *


class CB_TruncateSequence(Callback):
    """Progressively truncates sequence length during training using a scheduler.

    Args:
        truncate_length: maximum number of time steps to truncate
        scheduler: scheduling function controlling truncation over training
    """

    def __init__(self, truncate_length=50, scheduler=sched_ramp):
        self._truncate_length = truncate_length
        self._scheduler = scheduler

    def before_batch(self):
        if self.training:
            with torch.no_grad():
                ly = self.yb[0].shape[1]
                lim = int(self._scheduler(ly - self._truncate_length, 0, self.pct_train))
                if lim > 0:
                    # print(lx,ly,lim)
                    #         import pdb; pdb.set_trace()
                    self.learn.xb = tuple([x[:, :-lim] for x in self.xb])
                    self.learn.yb = tuple([y[:, :-lim] for y in self.yb])


class CB_AddLoss(Callback):
    """Adds an auxiliary loss to the minibatch after the primary loss is computed.

    Args:
        _loss_func: auxiliary loss function applied to predictions and targets
        alpha: scaling factor for the auxiliary loss
    """

    def __init__(self, _loss_func, alpha=1.0):
        self._loss_func = _loss_func
        self.alpha = alpha

    def after_loss(self):
        if not self.training:
            return

        loss = self.alpha * self._loss_func(self.pred, self.y)
        self.learn.loss_grad = loss + self.learn.loss_grad
        self.learn.loss = loss + self.learn.loss


class BatchLossFilter(Callback):
    """Selects the hardest samples in every batch representing a percentage of the total loss.

    Args:
        loss_perc: fraction of total loss to keep (1.0 keeps all samples)
        filter_criterion: per-sample loss function used for ranking
        schedule_func: optional function to adjust loss_perc over training
    """

    def __init__(
        self, loss_perc=1.0, filter_criterion=nn.HuberLoss(reduction="none"), schedule_func: Optional[callable] = None
    ):
        self.loss_perc = loss_perc
        self.filter_criterion = filter_criterion
        self.schedule_func = schedule_func

    def after_pred(self):
        """Selects hardest samples after model prediction and before loss computation."""
        if not self.training:
            return
        if self.schedule_func is None:
            loss_perc = self.loss_perc
        else:
            loss_perc = self.loss_perc * self.schedule_func(self.pct_train)
        if loss_perc == 1.0:
            return

        with torch.no_grad():
            losses = self.filter_criterion(self.pred, self.y)
            if losses.ndim >= 2:
                losses = losses.mean(tuple(range(1, losses.ndim)))
            losses /= losses.sum()

            idxs = torch.argsort(losses, descending=True)
            cut_idx = max(1, torch.argmax((losses[idxs].cumsum(0) > loss_perc).float()))
            idxs = idxs[:cut_idx]

        self.learn.xb = tuple(xbi[idxs] for xbi in self.learn.xb)
        self.learn.yb = tuple(ybi[idxs] for ybi in self.learn.yb)
        self.learn.pred = self.pred[idxs]


from fastai.callback.hook import *


@delegates()
class TimeSeriesRegularizer(HookCallback):
    """Adds activation regularization (AR) and temporal activation regularization (TAR) to the loss.

    Args:
        alpha: coefficient for AR penalty (L2 on activations)
        beta: coefficient for TAR penalty (L2 on consecutive activation differences)
        dim: time axis index; auto-detected from the hooked layer output if None
        detach: whether to detach the hooked output before computing penalties
        **kwargs: forwarded to HookCallback (must include modules)
    """

    run_before = TrainEvalCallback

    def __init__(self, alpha=0.0, beta=0.0, dim=None, detach=False, **kwargs):
        if "modules" not in kwargs:
            print("Warning: No module was provided to TimeSerieRegularizer")
        super().__init__(detach=detach, **kwargs)
        self.alpha = alpha
        self.beta = beta
        self.dim = dim

    def hook(self, m, i, o):
        #         import pdb; pdb.set_trace()
        if isinstance(o, torch.Tensor):
            self.out = o
        else:
            self.out = o[0]

        # find time axis if not already provided
        if self.dim is None:
            self.dim = np.argmax([0, self.out.shape[1], self.out.shape[2]])

    def after_loss(self):
        if not self.training:
            return

        h = self.out.float()

        if self.alpha != 0.0:
            l_a = float(self.alpha) * h.pow(2).mean()
            self.learn.loss_grad += l_a

        if self.beta != 0.0 and h.shape[self.dim] > 1:
            h_diff = (h[:, 1:] - h[:, :-1]) if self.dim == 1 else (h[:, :, 1:] - h[:, :, :-1])
            l_b = float(self.beta) * h_diff.pow(2).mean()
            self.learn.loss_grad += l_b


class ARInitCB(Callback):
    """Concatenates the target variable to the input for autoregression."""

    def before_fit(self):
        if hasattr(self.dls, "norm_stats"):
            n_u = len(self.dls.norm_stats.u.mean)
            n_inp = self.dls.one_batch()[0].shape[-1]
            if n_inp > n_u:
                raise ValueError(
                    "ARInitCB is incompatible with prediction-mode DataLoaders "
                    "(input already contains output features). "
                    "Use create_dls(..., prediction=False)."
                )

    def before_batch(self):
        x, y = self.xb[0], self.yb[0].as_subclass(type(self.xb[0]))
        self.learn.xb = (torch.cat((x, y), dim=-1),)


from matplotlib.lines import Line2D


def plot_grad_flow(named_parameters):
    """Plots the gradients flowing through different layers in the net during training.

    Can be used for checking for possible gradient vanishing / exploding problems.
    Call multiple times for transparent overlays representing the mean gradients.

    Args:
        named_parameters: iterator of (name, parameter) pairs from a model
    """
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n):
            layers.append(n)
            #             pdb.set_trace()
            ave_grads.append(0 if p.grad is None else p.grad.abs().mean().cpu())
            max_grads.append(0 if p.grad is None else p.grad.abs().max().cpu())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("Gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.yscale("log")
    plt.tight_layout()
    plt.legend(
        [Line2D([0], [0], color="c", lw=4), Line2D([0], [0], color="b", lw=4), Line2D([0], [0], color="k", lw=4)],
        ["max-gradient", "mean-gradient", "zero-gradient"],
    )


class CB_PlotGradient(Callback):
    """Plots the gradient distribution for every trainable parameter.

    Args:
        n_draws: number of gradient snapshots to plot across training
    """

    def __init__(self, n_draws=20):
        self.n_draws = n_draws

    def begin_fit(self):
        """Create a new figure to plot in."""
        plt.figure()
        plt.tight_layout()

    def after_backward(self):
        """Plot the gradient for every layer of the current minibatch."""
        # plotting n_draws times at the whole training
        if self.iter % (max(self.n_epoch * self.n_iter // self.n_draws, 1)) == 0:
            #         if self.iter == self.n_iter-1:
            plot_grad_flow(self.learn.model.named_parameters())


#             print('done')
