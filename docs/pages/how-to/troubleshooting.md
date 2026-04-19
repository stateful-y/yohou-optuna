# Troubleshooting

Solutions to common problems when using Yohou-Optuna.

## Installation Issues

**Problem: Package not found**
: Verify the package name: `pip install yohou_optuna` or `uv add yohou_optuna`.

**Problem: Import error after installation**
: Make sure you installed in the active environment: `python -c "import yohou_optuna; print(yohou_optuna.__version__)"`

## Search Issues

**Problem: `AttributeError: 'TPESampler' object has no attribute 'instantiate'`**
: You passed a raw Optuna object instead of a wrapper. Replace `sampler=optuna.samplers.TPESampler()` with `sampler=Sampler(sampler=optuna.samplers.TPESampler)`. The same applies to `Storage` and `Callback`. See [Configure OptunaSearchCV](configure.md).

**Problem: `TypeError: callbacks must be a dict of str to Callback`**
: You passed `callbacks` as a list instead of a dictionary. This error appears at `fit()` time, not at construction. Replace `callbacks=[Callback(...)]` with `callbacks={"name": Callback(...)}`. The dictionary keys are arbitrary names used for parameter routing.

**Problem: `ValueError: scoring parameter cannot be None`**
: The `scoring` parameter is required. Pass a scorer instance: `scoring=MeanAbsoluteError()`. For multi-metric search, pass a dict: `scoring={"mae": MeanAbsoluteError(), "mse": MeanSquaredError()}`.

**Problem: Results are not reproducible**
: Wrap the sampler with `Sampler` and pass `seed=`. Use `n_jobs=1` because parallel trial ordering is non-deterministic. See [Configure OptunaSearchCV](configure.md#choose-a-sampler).

**Problem: All trials return NaN**
: The forecaster may be failing silently. Set `error_score="raise"` to surface the underlying error. See [Handle Fitting Errors](configure.md#handle-fitting-errors).

**Problem: Search is slow**
: Reduce `n_trials` or set a `timeout` to cap execution time. Check that `n_splits` in your splitter is not too large. Consider using `n_jobs=-1` for parallel trial execution on a single machine, or distribute trials across multiple nodes with a shared database storage (see [Concepts: Parallelism](../explanation/concepts.md#parallelism)).

**Problem: CMA-ES sampler raises an error**
: CMA-ES does not support `CategoricalDistribution` parameters. Use `TPESampler` for mixed search spaces.

## Parameter Routing

**Problem: `ValueError: Invalid parameter ... for estimator`**
: The nested parameter path is wrong. For a parameter `alpha` on the `estimator` inside a `PointReductionForecaster`, use `"estimator__alpha"` (two underscores). Verify the full path with `forecaster.get_params(deep=True)`.

**Problem: `AttributeError` when cloning (`__init__` parameter not found)**
: This usually means a custom estimator or wrapper is missing a constructor parameter that `clone()` expects. Ensure all `__init__` parameters are stored as attributes with the same name.

**Problem: Fitting failures from invalid parameter values**
: By default, `error_score=np.nan` catches errors during cross-validation folds and records `NaN` for that trial. However, the best-found parameters are still used for refitting on the full dataset after all trials complete. If all trials failed, the refit step will raise. Check that your distribution ranges only produce valid values (e.g., `FloatDistribution(1e-4, 10.0)` instead of ranges that include negative or zero values for parameters like `alpha`).

## Cross-Validation Issues

**Problem: Optimistic CV scores that do not hold out-of-sample**
: Increase `n_splits` in your splitter, or switch to `ExpandingWindowSplitter` if you are using `SlidingWindowSplitter`. Ensure the forecast horizon in `cv` matches the horizon you will use at inference.

**Problem: `ValueError: Not enough data for the requested number of splits`**
: The training series is too short for the configured splitter. Reduce `n_splits` or `forecasting_horizon`, or use a shorter minimum training window.

## Scoring Issues

**Problem: `ValueError: scoring must be an instance of BaseScorer or a dict`**
: You passed `scoring` as a list. Only a single scorer or a dict of scorers is supported. Replace `scoring=[MeanAbsoluteError(), MeanSquaredError()]` with `scoring={"mae": MeanAbsoluteError(), "mse": MeanSquaredError()}`.

**Problem: `ValueError: For multi-metric scoring, the parameter refit must be set to a scorer key`**
: When `scoring` is a dict, set `refit` to one of the dictionary keys. For example, with `scoring={"mae": MeanAbsoluteError(), "mse": MeanSquaredError()}`, use `refit="mae"`. If you do not need refitting, set `refit=False`.

## Storage Issues

**Problem: Study not resuming from database**
: Make sure you use the same `study_name` and `Storage` configuration. See [Persist and Resume Studies](configure.md#persist-and-resume-studies).

**Problem: `OperationalError: unable to open database file`**
: The directory for the SQLite database does not exist. Create it first: `mkdir -p path/to/dir`.

## Getting Help

- [Open an issue on GitHub](https://github.com/stateful-y/yohou-optuna/issues/new)
- [Start a discussion](https://github.com/stateful-y/yohou-optuna/discussions)
