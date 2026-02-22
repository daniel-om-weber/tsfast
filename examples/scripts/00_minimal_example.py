# ---
# jupyter:
#   jupytext:
#     formats: notebooks//ipynb,scripts//py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %%
from tsfast.basics import *

dls = create_dls_silverbox(bs=16,win_sz=500,stp_sz=10)

# %%
lrn = RNNLearner(dls,rnn_type='lstm')
lrn.fit_flat_cos(n_epoch=1)

# %%
lrn.show_results(max_n=1)

# %%
