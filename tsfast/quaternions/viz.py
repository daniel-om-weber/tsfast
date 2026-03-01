"""Quaternion visualization utilities."""

__all__ = [
    "plot_scalar_inclination",
    "plot_quaternion_inclination",
    "plot_quaternion_rel_angle",
]

import torch

from .core import inclinationAngle, inclinationAngleAbs, rad2deg, relativeAngle


def plot_scalar_inclination(
    axs: list, in_sig: torch.Tensor, targ_sig: torch.Tensor, out_sig: torch.Tensor | None = None, **kwargs
):
    """Plot scalar inclination target, prediction, and error.

    Args:
        axs: list of matplotlib axes to plot on.
        in_sig: input signal tensor.
        targ_sig: target inclination tensor.
        out_sig: predicted inclination tensor, or None for batch display.
    """
    axs[0].plot(rad2deg(targ_sig).detach().cpu().numpy())
    axs[0].label_outer()
    axs[0].set_ylabel("inclination[deg]")

    if out_sig is not None:
        axs[0].plot(rad2deg(out_sig).detach().cpu().numpy())
        axs[0].legend(["y", "y-hat"])
        axs[1].plot(rad2deg(targ_sig - out_sig).detach().cpu().numpy())
        axs[1].label_outer()
        axs[1].set_ylabel("error[deg]")

    axs[-1].plot(in_sig)


def plot_quaternion_inclination(
    axs: list, in_sig: torch.Tensor, targ_sig: torch.Tensor, out_sig: torch.Tensor | None = None, **kwargs
):
    """Plot quaternion inclination target, prediction, and error.

    Args:
        axs: list of matplotlib axes to plot on.
        in_sig: input signal tensor.
        targ_sig: target quaternion tensor.
        out_sig: predicted quaternion tensor, or None for batch display.
    """
    axs[0].plot(rad2deg(inclinationAngleAbs(targ_sig)).detach().cpu().numpy())
    axs[0].label_outer()
    axs[0].legend(["y"])
    axs[0].set_ylabel("inclination[deg]")

    if out_sig is not None:
        axs[0].plot(rad2deg(inclinationAngleAbs(out_sig)).detach().cpu().numpy())
        axs[0].legend(["y", "y-hat"])
        axs[1].plot(rad2deg(inclinationAngle(out_sig, targ_sig)).detach().cpu().numpy())
        axs[1].label_outer()
        axs[1].set_ylabel("error[deg]")
        if "ref" in kwargs:
            axs[1].plot(rad2deg(inclinationAngle(targ_sig, kwargs["ref"])).detach().cpu().numpy())
            axs[1].legend(["y-hat", "y_ref"])

    axs[-1].plot(in_sig)


def plot_quaternion_rel_angle(
    axs: list, in_sig: torch.Tensor, targ_sig: torch.Tensor, out_sig: torch.Tensor | None = None, **kwargs
):
    """Plot relative quaternion angle target, prediction, and error.

    Args:
        axs: list of matplotlib axes to plot on.
        in_sig: input signal tensor.
        targ_sig: target quaternion tensor.
        out_sig: predicted quaternion tensor, or None for batch display.
    """
    first_targ = targ_sig[0].repeat(targ_sig.shape[0], 1)
    axs[0].plot(rad2deg(relativeAngle(first_targ, targ_sig)).detach().cpu().numpy())
    axs[0].label_outer()
    axs[0].legend(["y"])
    axs[0].set_ylabel("angle[deg]")

    if out_sig is not None:
        axs[0].plot(rad2deg(relativeAngle(first_targ, out_sig)).detach().cpu().numpy())
        axs[0].legend(["y", "y-hat"])
        axs[1].plot(rad2deg(relativeAngle(out_sig, targ_sig)).detach().cpu().numpy())
        axs[1].label_outer()
        axs[1].set_ylabel("error[deg]")

    axs[-1].plot(in_sig)
