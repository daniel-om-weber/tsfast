__all__ = ["ARProg"]

from ..data import *
from ..models import *
from ..learner import *
from fastai.basics import *


class ARProg(nn.Module):
    @delegates(RNN, keep=True)
    def __init__(self, n_u, n_x, n_y, init_sz, **kwargs):
        super().__init__()
        store_attr()

        # the output is n_x+n_y even if we output only n_y, because we need the output external states back into the model
        self.rnn_model = AR_Model(
            SimpleRNN(input_size=n_u + n_x + n_y, output_size=n_x + n_y, return_state=True, **kwargs),
            model_has_state=True,
            return_state=True,
            ar=True,
            out_sz=n_x + n_y,
        )

    def forward(self, x):
        y_x = x[..., self.n_u :]  # measured output and external state
        u = x[..., : self.n_u]  # measured input

        out_init, h = self.rnn_model(u[:, : self.init_sz], y_x[:, : self.init_sz], ar=False)
        self.rnn_model.y_init = y_x[:, self.init_sz : self.init_sz + 1]
        out_prog, _ = self.rnn_model(u[:, self.init_sz :], h_init=h, ar=True)

        result = torch.cat([out_init, out_prog], 1)

        return result[..., -self.n_y :]
