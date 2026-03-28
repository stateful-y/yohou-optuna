# Troubleshooting

Solutions to common problems when using Yohou-Optuna.

## Fitting and Import Issues

**Problem: `ImportError` after installation**
: Verify you installed in the active environment: `python -c "import yohou_optuna; print(yohou_optuna.__version__)"`

**Problem: `TypeError: cannot pickle 'TPESampler' object` during `fit()`**
: You passed a raw Optuna object instead of a wrapper. Replace `sampler=optuna.samplers.TPESampler()` with `sampler=Sampler("TPESampler")`. The same applies to `Storage` and `Callback`. See [Configure OptunaSearchCV](configure.md).

**Problem: `AttributeError` when cloning - `__init__` parameter not found**
: This usually means a parameter name in `param_distributions` does not match the forecaster's API. Check spelling and use `forecaster.get_params()` to list valid parameter names.

## Parameter Routing

**Problem: `ValueError: Invalid parameter ... for estimator`**
: The nested parameter path is wrong. For a parameter `alpha` on the `regressor` inside a `PointReductionForecaster`, use `"regressor__alpha"` - two underscores. Verify the full path with `forecaster.get_params(deep=True)`.

**Problem: Parameter distribution values cause fitting failures**
: Set `error_score=float("nan")` to assign a penalty score for failed trials rather than aborting the search. Then inspect `cv_results_` for trials with `nan` scores to narrow down the bad region.

## Cross-Validation Issues

**Problem: Optimistic CV scores that do not hold out-of-sample**
: Increase `n_splits` in your splitter, or switch to `ExpandingWindowSplitter` if you are using `SlidingWindowSplitter`. Ensure the forecast horizon in `cv` matches the horizon you will use at inference.

**Problem: `ValueError: Not enough data for the requested number of splits`**
: The training series is too short for the configured splitter. Reduce `n_splits` or `forecasting_horizon`, or use a shorter minimum training window.

## Study Persistence

**Problem: `Study not found` when resuming**
: The `study_name` must match exactly between `fit()` calls. Check that the storage URL points to the same file or database.

**Problem: `OperationalError: unable to open database file`**
: The directory for the SQLite database does not exist. Create it first: `mkdir -p path/to/dir`.

## Scoring Issues

**Problem: `ValueError: refit must be a string matching a scorer name` with multi-metric search**
: When `scoring` is a list or dict, set `refit` to the string name of the metric to use for selecting the best forecaster. For example: `refit="MeanAbsoluteError"`.

## Getting Help

- [Open an issue on GitHub](https://github.com/stateful-y/yohou-optuna/issues/new)
- [Start a discussion](https://github.com/stateful-y/yohou-optuna/discussions)
