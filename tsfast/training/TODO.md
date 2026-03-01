# training/core.py — remaining cleanup

## Add a Protocol type for `dls`
`dls` is duck-typed for `.train`, `.valid`, `.test` with no type hint. A `Protocol` class would make the API self-documenting and give IDE support.
