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
#     display_name: python3
#     language: python
#     name: python3
# ---

# %%
from tsfast.basics import *

# %%
external_datasets_simulation

# %%
dls = external_datasets_simulation[0]()

# %%
dls.show_batch()

# %%
from fastai.basics import default_device
default_device(True)

# %%
dls = create_dls_wh(cached=True,num_workers=0)

# %%
lrn = RNNLearner(dls,n_skip=50,hidden_size=100,rnn_type='lstm')

# %%
lrn.fit_flat_cos(1,lr=3e-3,pct_start=0.3)

# %%
for dl_func in external_datasets_simulation:
    dls = dl_func()
    lrn = RNNLearner(dls,n_skip=50,rnn_type='lstm')
    lrn.fit_flat_cos(10,lr=3e-3,pct_start=0.3)

# %%
for dl_func in external_datasets_simulation:
    dls = dl_func()
    lrn = RNNLearner(dls,n_skip=50)
    lrn.fit_flat_cos(10,lr=3e-3,pct_start=0.3)

# %%
dls = create_dls_silverbox('silverbox')
lrn = RNNLearner(dls,n_skip=50)
lrn.fit_flat_cos(10,lr=3e-3,pct_start=0.3)

# %%
lrn.validate(2)

# %%
preds = lrn.get_preds(2)

# %%
preds[0][0]

# %%
from fastai.basics import *
plt.figure()
plt.plot(preds[0][1]-preds[1][1])

# %%
# %matplotlib widget

# %%
lrn.show_results(2)

# %%
lrn.show_results(2)

# %%
