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
import identibench as idb

dls = create_dls_cascaded_tanks().cpu()

# %%
lrn = FranSysLearner(dls,idb.BenchmarkCascadedTanks_Simulation.init_window,attach_output=True,hidden_size=40)
#lrn.add_cb(FranSysCallback([lrn.model]))
lrn.add_cb(TimeSeriesRegularizer(3,3,modules=[lrn.model.rnn_prognosis,lrn.model.rnn_diagnosis]))
lrn.fit_flat_cos(n_epoch=50)

# %%
lrn = FranSysLearner(dls,idb.BenchmarkCascadedTanks_Simulation.init_window,attach_output=True,hidden_size=40)
#lrn.add_cb(FranSysCallback([lrn.model]))
lrn.add_cb(TimeSeriesRegularizer(6,6,modules=[lrn.model.rnn_prognosis,lrn.model.rnn_diagnosis]))
lrn.fit_flat_cos(n_epoch=50)

# %%
lrn = FranSysLearner(dls,idb.BenchmarkCascadedTanks_Simulation.init_window,attach_output=True,hidden_size=40)
#lrn.add_cb(FranSysCallback([lrn.model]))
lrn.add_cb(TimeSeriesRegularizer(6,6,modules=[lrn.model.rnn_prognosis,lrn.model.rnn_diagnosis]))
lrn.fit_flat_cos(n_epoch=50)

# %%
lrn = FranSysLearner(dls,idb.BenchmarkCascadedTanks_Simulation.init_window,attach_output=True,hidden_size=100)
#lrn.add_cb(FranSysCallback([lrn.model]))
lrn.add_cb(TimeSeriesRegularizer(9,9,modules=[lrn.model.rnn_prognosis,lrn.model.rnn_diagnosis]))
lrn.fit_flat_cos(n_epoch=50)

# %%
lrn.show_results(ds_idx=-1,max_n=1)

# %%
lrn.show_results(ds_idx=-1,max_n=1)

# %%
lrn.validate(ds_idx=-1)

# %%
