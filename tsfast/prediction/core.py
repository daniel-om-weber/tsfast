"""Prediction callbacks for autoregressive input concatenation."""

__all__ = ["PredictionCallback"]

import torch
from fastai.callback.core import Callback


class PredictionCallback(Callback):
    """Concatenates system output to input for autoregression.

    Assumes a 1-tuple as input. Must execute first so downstream callbacks
    see the correct data.

    Args:
        t_offset: number of steps the output is shifted in the past, shortens
            the sequence length by that amount
    """

    order = -56

    def __init__(
        self,
        t_offset: int = 1,
    ):
        super().__init__()
        self.t_offset = t_offset

    def before_batch(self):
        # output has to be casted to the input tensor type
        x = self.x
        y = self.yb[0].as_subclass(type(x))

        if self.t_offset != 0:
            x = x[:, self.t_offset :, :]
            y = y[:, : -self.t_offset, :]

            # shorten the output by the same size
            self.learn.yb = tuple((y[:, self.t_offset :, :] for y in self.yb))

        # concatenate and reconvert to tuple
        self.learn.xb = (torch.cat((x, y), dim=-1),)
