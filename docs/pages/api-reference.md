# API Reference

Complete API reference for Yohou-Optuna.

## OptunaSearchCV

::: yohou_optuna.search.OptunaSearchCV
    options:
      show_root_heading: true
      show_source: true
      show_bases: true
      inherited_members: true
      filters:
        - "!^_"

## Optuna wrappers

Yohou-Optuna re-exports the following wrapper classes for Optuna compatibility
with sklearn's `clone()` and serialization:

- `Sampler`
- `Storage`
- `Callback`

See [sklearn-optuna](https://github.com/stateful-y/sklearn-optuna) for detailed
API references on these wrappers.
