# training/core.py — remaining cleanup

## Move `_make_flat_cos_scheduler` to `schedulers.py`
`schedulers.py` already has `sched_lin_p` and `sched_ramp`. The flat-cosine scheduler (core.py lines ~41-51) is stranded in core.py for no reason.

## Vectorize per-sample loss in `get_worst`
Currently computes loss in a Python loop one sample at a time. Use `reduction='none'` to vectorize:
```python
# instead of:
per_sample = torch.tensor([loss_func(preds[i:i+1], ...).item() for i in range(len(preds))])
# use something like:
per_sample = loss_func_no_reduce(preds, targs)  # needs reduction='none' variant
```

## Add a Protocol type for `dls`
`dls` is duck-typed for `.train`, `.valid`, `.test` with no type hint. A `Protocol` class would make the API self-documenting and give IDE support.
