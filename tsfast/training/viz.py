"""Visualization utilities for training inspection."""

__all__ = [
    "plot_sequence",
    "plot_seqs_single_figure",
    "plot_seqs_multi_figures",
    "layout_samples",
    "plot_grad_flow",
    "grad_norm",
]

from collections.abc import Callable, Iterator

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from torch import Tensor


def plot_sequence(axs: list, in_sig: Tensor, targ_sig: Tensor, out_sig: Tensor | None = None, **kwargs):
    """Plot input, target, and optional prediction sequences on subplot axes."""
    signal_names = kwargs.pop("signal_names", None)
    u_names, y_names = signal_names if signal_names is not None else (None, None)

    n_targ = targ_sig.shape[1]
    n_out = out_sig.shape[1] if out_sig is not None else n_targ
    n_ax = len(axs) - 1

    for j in range(min(n_ax, max(n_targ, n_out))):
        ax = axs[j]
        if j < n_targ:
            ax.plot(targ_sig[:, j], label="y", alpha=0.7)
        if out_sig is not None and j < n_out:
            label = "ŷ" if j < n_targ else "ŷ (aux)"
            ax.plot(out_sig[:, j], label=label, alpha=0.7)
        if "ref" in kwargs and j < kwargs["ref"].shape[1]:
            ax.plot(kwargs["ref"][:, j], label="ref", alpha=0.5)
        if j >= n_targ:
            ax.set_title(f"Channel {j} (auxiliary)", fontsize=10)
        elif y_names is not None and j < len(y_names):
            ax.set_title(y_names[j], fontsize=10)
        ax.legend(fontsize=8)
        ax.label_outer()

    if u_names is not None:
        for k, name in enumerate(u_names):
            if k < in_sig.shape[1]:
                axs[-1].plot(in_sig[:, k], label=name)
        axs[-1].legend(fontsize=8)
    else:
        axs[-1].plot(in_sig)


def plot_seqs_single_figure(
    n_samples: int, n_targ: int, samples: list, plot_func: Callable, outs: list | None = None, **kwargs
):
    """Plot multiple sample sequences in a single figure grid."""
    rows = max(1, ((n_samples - 1) // 3) + 1)
    cols = min(3, n_samples)
    fig = plt.figure(figsize=(9, 2 * cols))
    outer_grid = fig.add_gridspec(rows, cols)
    for i in range(n_samples):
        in_sig = samples[i][0]
        targ_sig = samples[i][1]
        out_sig = outs[i][0] if outs is not None else None
        inner_grid = outer_grid[i].subgridspec(n_targ + 1, 1)
        axs = [fig.add_subplot(inner_grid[j]) for j in range(n_targ + 1)]
        plot_func(axs, in_sig, targ_sig, out_sig=out_sig, **kwargs)
    plt.tight_layout()


def plot_seqs_multi_figures(
    n_samples: int, n_targ: int, samples: list, plot_func: Callable, outs: list | None = None, **kwargs
):
    """Plot each sample sequence in its own separate figure."""
    for i in range(n_samples):
        fig = plt.figure(figsize=(9, 3))
        axs = fig.subplots(nrows=n_targ + 1, sharex=True)
        in_sig = samples[i][0]
        targ_sig = samples[i][1]
        out_sig = outs[i][0] if outs is not None else None
        plot_func(axs, in_sig, targ_sig, out_sig=out_sig, **kwargs)
        plt.tight_layout()


def layout_samples(n_samples: int, n_targ: int, samples: list, plot_func: Callable, outs: list | None = None, **kwargs):
    """Dispatch to single or multi figure layout based on sample count."""
    if n_samples > 3:
        plot_seqs_single_figure(n_samples, n_targ, samples, plot_func, outs, **kwargs)
    else:
        plot_seqs_multi_figures(n_samples, n_targ, samples, plot_func, outs, **kwargs)


def plot_grad_flow(named_parameters: Iterator) -> None:
    """Plot gradient flow through network layers.

    Useful for checking gradient vanishing/exploding. Call multiple times
    for transparent overlays representing the mean gradients.

    Args:
        named_parameters: iterator of (name, parameter) pairs from a model
    """
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n):
            layers.append(n)
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


def grad_norm(parameters) -> float:
    """Compute the total gradient norm across all parameters.

    Args:
        parameters: iterable of model parameters
    """
    grads = [p.grad.detach().flatten() for p in parameters if p.grad is not None]
    if not grads:
        return 0.0
    return torch.cat(grads).norm().item()
