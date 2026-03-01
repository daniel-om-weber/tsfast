# Training module code quality improvements

## Remaining improvements

- **`FranSysRegularizer.__call__`** (`aux_losses.py`): ~75-line method doing 4 independent things (state sync, diagnosis loss, OSP loss, TAR loss). The `sync_type` dispatch is a 6-branch elif chain that could be a dict mapping. Each loss term could be a private method.
