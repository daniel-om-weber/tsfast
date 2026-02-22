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


# %%
#build_model function using the IdentiBench API
def build_model(context: idb.TrainingContext):
    #dataloader from benchmark speciication
    dls = create_dls_from_spec(context.spec)
    #RNN Learner with provided configuration
    lrn = RNNLearner(dls, 
                     rnn_type=context.hyperparameters['model_type'], 
                     num_layers=context.hyperparameters['num_layers'], 
                     n_skip=context.spec.init_window)
    #training with 
    lrn.fit_flat_cos(n_epoch=1)
    model = InferenceWrapper(lrn)
    return model


# %%
# configure model and benchmarks
model_config = {'model_type':'lstm','num_layers':1}
benchmarks = idb.workshop_benchmarks.values()
# test identification method on all provided benchmark datasets
results = idb.run_benchmarks(benchmarks,build_model,model_config)

# %%
results

# %%
