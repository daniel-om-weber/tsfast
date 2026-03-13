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

# %% [markdown]
# # Example 18: Saving and Loading
#
# After training a model, you need to save it -- either for deployment (run
# inference without retraining) or to resume training later. TSFast provides
# three approaches at different levels of convenience:
#
# 1. **`save_model` / `load_model`** -- save the model for inference
# 2. **`save` / `load`** -- save the entire Learner to resume training
# 3. **`save_checkpoint` / `load_checkpoint`** -- fallback when the Learner
#    contains unpicklable components (e.g. lambda loss functions)

# %% [markdown]
# ## Prerequisites
#
# This notebook builds on Example 00 (training basics). Make sure TSFast is
# installed:
#
# ```bash
# uv sync --extra dev
# ```

# %% [markdown]
# ## Setup

# %%
import tempfile
from pathlib import Path

import torch
from tsfast.inference import load_model
from tsfast.tsdata.benchmark import create_dls_silverbox
from tsfast.training import Learner, RNNLearner, fun_rmse

# %%
tmpdir = Path(tempfile.mkdtemp())

# %% [markdown]
# ## Train a Quick Model
#
# We train a small LSTM on the Silverbox benchmark. This gives us a trained
# Learner to demonstrate all three save/load approaches.

# %%
dls = create_dls_silverbox(bs=16, win_sz=500, stp_sz=10)
lrn = RNNLearner(dls, rnn_type='lstm', hidden_size=40, metrics=[fun_rmse])
lrn.fit_flat_cos(n_epoch=3)

# %% [markdown]
# ---
# ## Part 1: Save/Load Model for Inference
#
# Use `save_model` when you only need to run the trained model for inference.
# It saves the full model object (including normalization layers) via
# `torch.save`. Loading returns a standalone `nn.Module` -- no Learner needed.

# %%
lrn.save_model(tmpdir / 'model.pth')

# %% [markdown]
# `load_model` is a standalone function in `tsfast.inference` that returns
# the model in eval mode, ready for inference.

# %%
model = load_model(tmpdir / 'model.pth')
print(type(model))

# %% [markdown]
# Run inference directly with the loaded model. The normalization layers are
# included, so you can pass raw data.

# %%
xb, yb = next(iter(dls.valid))
xb = xb.to(lrn.device)

with torch.no_grad():
    pred = model(xb)
    if isinstance(pred, tuple):
        pred = pred[0]

print(f"Input:      {xb.shape}")
print(f"Prediction: {pred.shape}")

# %% [markdown]
# Verify that the loaded model produces the same predictions as the original.

# %%
preds_orig, _ = lrn.get_preds()

lrn_loaded = RNNLearner(dls, rnn_type='lstm', hidden_size=40)
lrn_loaded.model = model
preds_loaded, _ = lrn_loaded.get_preds()

max_diff = (preds_orig - preds_loaded).abs().max().item()
print(f"Max prediction difference: {max_diff:.2e}")
assert max_diff < 1e-5, "Predictions should match!"

# %% [markdown]
# ---
# ## Part 2: Save/Load Learner for Training Resume
#
# Use `save` / `load` when you want to resume training later. This pickles the
# entire Learner state (model, optimizer, recorder, hyperparameters) except for
# `dls`, which you re-provide when loading.

# %%
lrn.save(tmpdir / 'learner.pth')

# %% [markdown]
# `load` is a **classmethod**. The only argument besides the path is `dls` --
# everything else (model architecture, optimizer state, training history) is
# restored from the saved file.

# %%
lrn2 = Learner.load(tmpdir / 'learner.pth', dls=dls)

# %% [markdown]
# The recorder history is preserved -- our 3 training epochs are still there.

# %%
print(f"Recorder entries after load: {len(lrn2.recorder)}")
assert len(lrn2.recorder) == 3

# %% [markdown]
# Now resume training for 3 more epochs. The recorder continues from where we
# left off.

# %%
lrn2.fit_flat_cos(n_epoch=3)

# %%
print(f"Total recorder entries: {len(lrn2.recorder)}")
assert len(lrn2.recorder) == 6  # 3 original + 3 resumed

# %% [markdown]
# The validation loss should be lower after the additional training.

# %%
val_before = lrn2.recorder[2][1]  # last epoch before resume
val_after = lrn2.recorder[-1][1]  # last epoch after resume
print(f"Val loss before resume: {val_before:.4f}")
print(f"Val loss after resume:  {val_after:.4f}")

# %% [markdown]
# This approach also works for `TbpttLearner` -- the subclass type and all
# attributes (like `sub_seq_len`) are preserved automatically.

# %% [markdown]
# ---
# ## Part 3: Checkpoint Fallback
#
# If `save()` fails because the Learner contains components that can't be
# pickled (e.g. lambda loss functions), use `save_checkpoint` /
# `load_checkpoint` instead. This saves only the model weights, optimizer
# state, and recorder -- you must reconstruct the Learner manually.

# %%
lrn.save_checkpoint(tmpdir / 'checkpoint.pth')

# %% [markdown]
# To load a checkpoint, first create a new Learner with the same architecture
# and configuration, then call `load_checkpoint`.

# %%
lrn3 = RNNLearner(dls, rnn_type='lstm', hidden_size=40, metrics=[fun_rmse])
lrn3.load_checkpoint(tmpdir / 'checkpoint.pth')

# %%
print(f"Recorder entries after checkpoint load: {len(lrn3.recorder)}")
assert len(lrn3.recorder) == 3

# %% [markdown]
# Verify the checkpoint restored the same model weights.

# %%
preds_ckpt, _ = lrn3.get_preds()
max_diff = (preds_orig - preds_ckpt).abs().max().item()
print(f"Max prediction difference: {max_diff:.2e}")
assert max_diff < 1e-5, "Checkpoint should restore identical weights!"

# %% [markdown]
# Resume training from the checkpoint -- the optimizer state is restored when
# `fit()` is called.

# %%
lrn3.fit_flat_cos(n_epoch=3)
print(f"Total recorder entries: {len(lrn3.recorder)}")
assert len(lrn3.recorder) == 6

# %% [markdown]
# ## Key Takeaways
#
# | Method | What it saves | When to use |
# |--------|--------------|-------------|
# | `save_model` / `load_model` | Full model object | Inference / deployment (`from tsfast.inference import load_model`) |
# | `save` / `load` | Entire Learner (minus dls) | Resume training (default) |
# | `save_checkpoint` / `load_checkpoint` | Weights + optimizer + recorder | Resume training with unpicklable components |
#
# - **Start with `save` / `load`** -- it's the simplest approach for resuming
#   training.
# - **Use `save_model`** when you only need the model for inference and don't
#   need the training loop.
# - **Fall back to `save_checkpoint`** only if `save()` raises a pickling
#   error (e.g. lambda loss functions).
