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
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Physics-Informed Neural Networks (PINN)
#
# This example demonstrates how to train physics-informed RNNs for system identification
# using the spring-damper system: $ma + cv + kx = u$
#
# Two approaches are shown:
# 1. **Basic RNN** with collocation points — physics-only training (surrogate model)
# 2. **PIRNN** — combines data fitting with physics constraints and supports variable initial conditions

# %% [markdown]
# ## Setup

# %%
from tsfast.basics import *
from fastai.basics import *
from tsfast.pinn.core import *

# Physical parameters (must match dataset)
MASS = 1.0
SPRING_CONSTANT = 1.0
DAMPING_COEFFICIENT = 0.1
DT = 0.01

def spring_damper_physics(u, y_pred, y_ref):
    "Physics loss for spring-damper: ma + cv + kx = u"
    x, v = y_pred[:, :, 0], y_pred[:, :, 1]
    u_force = u[:, :, 0]
    a = diff1_forward(v, DT)
    dx_dt = diff1_forward(x, DT)

    loss = {
        'physics': ((MASS * a + DAMPING_COEFFICIENT * v + SPRING_CONSTANT * x - u_force) ** 2).mean(),
        'derivative': ((v - dx_dt) ** 2).mean(),
    }
    # Initial condition loss when reference data is available
    if y_ref is not None:
        init_sz = 10
        loss['initial'] = ((x[:, :init_sz] - y_ref[:, :init_sz, 0]) ** 2).mean()
    return loss


# %%
path = Path("../test_data/pinn")
dls = create_dls(
    u=['u'], y=['x', 'v'],
    dataset=path,
    win_sz=100, stp_sz=1, valid_stp_sz=1,
    bs=32, n_batches_train=300
).cpu()

# %% [markdown]
# ## Approach 1: Basic RNN with Collocation Points
#
# Train a standard RNN using only physics constraints (no data fitting).
# Collocation points are randomly generated excitation signals that the model
# must satisfy the physics equations on — useful for surrogate models of known ODEs.

# %%
learn = RNNLearner(
    dls, rnn_type='lstm', num_layers=1, hidden_size=10,
    loss_func=zero_loss, metrics=[fun_rmse]
)

learn.add_cb(CollocationPointsCB(
    norm_input=dls.train.after_batch[0],
    generate_pinn_input=lambda bs, sl, dev: generate_excitation_signals(
        bs, sl, n_inputs=1, dt=DT, device=dev,
        amplitude_range=(0.5, 2.0), frequency_range=(0.1, 3.0)
    ),
    physics_loss_func=spring_damper_physics,
    weight=1.0
))

learn.fit_flat_cos(10, 3e-3)

# %%
learn.show_results(max_n=3, ds_idx=1)

# %% [markdown]
# ## Approach 2: PIRNN with Data + Physics
#
# PIRNN (Physics-Informed RNN) combines data fitting with physics constraints.
# It uses a `StateEncoder` to initialize hidden states from observed initial conditions,
# enabling the model to handle variable initial conditions at inference time.
#
# - `PhysicsLossCallback`: enforces physics on training data batches
# - `CollocationPointsCB`: enforces physics on randomly generated inputs (generalisation)

# %%
from tsfast.pinn.pirnn import PIRNNLearner

learn = PIRNNLearner(
    dls, init_sz=10, attach_output=True,
    rnn_type='gru', rnn_layer=1, hidden_size=20,
    state_encoder_hidden=32,
    loss_func=zero_loss, metrics=[fun_rmse]
)

# Physics on training data
learn.add_cb(PhysicsLossCallback(
    norm_input=dls.train.after_batch[0],
    physics_loss_func=spring_damper_physics,
    weight=1.0,
    loss_weights={'physics': 1.0, 'derivative': 1.0, 'initial': 10.0},
    n_inputs=1
))

# Physics on collocation points with StateEncoder initialization
learn.add_cb(CollocationPointsCB(
    norm_input=dls.train.after_batch[0],
    generate_pinn_input=lambda bs, sl, dev: generate_excitation_signals(
        bs, sl, n_inputs=1, dt=DT, device=dev,
        amplitude_range=(0.5, 2.0), frequency_range=(0.1, 3.0)
    ),
    physics_loss_func=spring_damper_physics,
    weight=0.5,
    init_mode='state_encoder',
    output_ranges=[(-1.0, 1.0), (-2.0, 2.0)]
))

learn.fit_flat_cos(50, 3e-3)

# %%
learn.show_results(max_n=3, ds_idx=1)
