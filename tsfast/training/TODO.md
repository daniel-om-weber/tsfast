# Training module code quality improvements

## Remaining improvements

- **Naming inconsistency**: `SkipNLoss`, `CutLoss`, `NormLoss`, `RandSeqLenLoss` are PascalCase but are functions returning closures (should be snake_case). `add_loss`, `physics_loss`, `transition_smoothness`, `consistency_loss` are classes but use snake_case (should be PascalCase). Renaming is a breaking API change — needs deprecation aliases or a coordinated rename.
- **`FranSysRegularizer.__call__`** (`aux_losses.py`): ~75-line method doing 4 independent things (state sync, diagnosis loss, OSP loss, TAR loss). The `sync_type` dispatch is a 6-branch elif chain that could be a dict mapping. Each loss term could be a private method.
- **`ignore_nan` fragility** (`losses.py`): Silently assumes `args[-1]` is the target and only checks `[..., -1]` (last feature dim) for NaN. Could use explicit parameters or at minimum assert the shape expectation.
