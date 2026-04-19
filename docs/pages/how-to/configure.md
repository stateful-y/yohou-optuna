# How to Configure OptunaSearchCV

This guide shows you how to configure `OptunaSearchCV` for common search scenarios: changing the sampler, adding callbacks, persisting studies, and customizing cross-validation.

## Prerequisites

- Yohou-Optuna installed ([Getting Started](../tutorials/getting-started.md))
- Familiarity with `OptunaSearchCV.fit()` basics

## Choose a Sampler

The `sampler` parameter controls the optimization strategy. The default (TPE) works well for most cases. Wrap any Optuna sampler with the `Sampler` class to make it compatible with `get_params()` / `set_params()` / `clone()`.

```python
import optuna
from yohou_optuna import OptunaSearchCV, Sampler

search = OptunaSearchCV(
    forecaster=forecaster,
    param_distributions=distributions,
    scoring=scorer,
    n_trials=50,
    sampler=Sampler(sampler=optuna.samplers.TPESampler, seed=42),
)
```

For other strategies, swap the sampler class:

```python
# CMA-ES: effective for continuous spaces with correlated parameters
sampler=Sampler(sampler=optuna.samplers.CmaEsSampler, seed=42)

# Gaussian Process: best for very small budgets (< 20 trials)
sampler=Sampler(sampler=optuna.samplers.GPSampler)

# Random: useful as a baseline or for reproducible ablations
sampler=Sampler(sampler=optuna.samplers.RandomSampler, seed=42)
```

Pass `seed` for reproducible results when `n_jobs=1`. Always use the `Sampler` wrapper rather than a raw Optuna sampler object because raw Optuna objects are not compatible with `clone()`.

!!! tip
    Start with TPE. Switch to CMA-ES only when you have a large all-continuous search space and notice slow convergence.

## Add Callbacks

Callbacks run after each completed trial. Use them for early stopping, logging, or custom logic. Pass a dictionary mapping callback names to `Callback` instances:

```python
from optuna.study import MaxTrialsCallback
from yohou_optuna import Callback

search = OptunaSearchCV(
    forecaster=forecaster,
    param_distributions=distributions,
    scoring=scorer,
    n_trials=200,
    callbacks={
        "stop": Callback(callback=MaxTrialsCallback, n_trials=50),
    },
)
```

`MaxTrialsCallback` stops the study once the specified number of trials completes, regardless of the `n_trials` setting on `OptunaSearchCV`. This is useful when you want to set a generous upper bound on trials but stop early once you have enough results.

Always use the `Callback` wrapper instead of a raw Optuna callback for the same cloneability reasons as `Sampler`.

## Write a Custom Callback

Any callable class that accepts `study` and `trial` arguments works as a callback:

```python
class EarlyStoppingCallback:
    def __init__(self, patience: int = 10):
        self.patience = patience

    def __call__(self, study, trial):
        if trial.number >= self.patience:
            best = study.best_trial.number
            if trial.number - best >= self.patience:
                study.stop()

search = OptunaSearchCV(
    forecaster=forecaster,
    param_distributions=distributions,
    scoring=scorer,
    n_trials=200,
    callbacks={
        "early_stop": Callback(callback=EarlyStoppingCallback, patience=10),
    },
)
```

## Persist and Resume Studies

For long-running or distributed searches, save the study to a storage backend:

```python
import optuna
from yohou_optuna import Storage

search = OptunaSearchCV(
    forecaster=forecaster,
    param_distributions=distributions,
    scoring=scorer,
    n_trials=50,
    storage=Storage(storage=optuna.storages.RDBStorage, url="sqlite:///my_study.db"),
)

search.fit(y_train, forecasting_horizon=12)
```

To add more trials later, pass the existing study back to `fit()`:

```python
search.n_trials = 50  # 50 additional trials
search.fit(y_train, forecasting_horizon=12, study=search.study_)
```

To name a study for easier identification, create it externally and pass it via `fit()`:

```python
study = optuna.create_study(
    study_name="ridge_air_passengers",
    direction="maximize",
    storage="sqlite:///my_study.db",
)
search.fit(y_train, forecasting_horizon=12, study=study)
```

## Use a Custom CV Splitter

By default, `OptunaSearchCV` uses a 5-fold expanding window. Pass any Yohou splitter to change this:

```python
from yohou.model_selection import SlidingWindowSplitter

search = OptunaSearchCV(
    forecaster=forecaster,
    param_distributions=distributions,    scoring=scorer,    n_trials=30,
    cv=SlidingWindowSplitter(n_splits=5, train_size=24),
)
```

Use `ExpandingWindowSplitter` (default) for growing training windows. Use `SlidingWindowSplitter` for a fixed-size training window.

## Handle Fitting Errors

By default, `error_score=np.nan` catches fitting errors during cross-validation folds and records `NaN` for that trial. To stop the search immediately on the first error instead, set `error_score="raise"`:

```python
search = OptunaSearchCV(
    forecaster=forecaster,
    param_distributions=distributions,
    scoring=scorer,
    n_trials=50,
    error_score="raise",  # stop on first error (useful during development)
)
```

Use `error_score="raise"` during development to catch bad parameter combinations early. In production, keep the default (`np.nan`) so the search continues past occasional failures.

## Filter Failed Trials

After fitting, inspect which trials failed:

```python
import polars as pl

results = pl.DataFrame(search.cv_results_)
failed = results.filter(pl.col("mean_test_score").is_nan())
print(f"{len(failed)} trials failed out of {len(results)}")
```

## Collect Training Scores

To compute scores on the training folds in addition to the validation folds, set `return_train_score=True`:

```python
search = OptunaSearchCV(
    forecaster=forecaster,
    param_distributions=distributions,
    scoring=scorer,
    n_trials=30,
    return_train_score=True,
)

search.fit(y_train, forecasting_horizon=12)

# cv_results_ now contains train_score columns alongside test_score columns
import polars as pl
results = pl.DataFrame(search.cv_results_)
print(results.select(["params", "mean_test_score", "mean_train_score"]))
```

Large gaps between training and test scores suggest overfitting.

## See Also

- [About OptunaSearchCV](../explanation/concepts.md): understand samplers, temporal CV, and wrapper classes
- [Multi-Metric Search](multi-metric-search.md): evaluate multiple metrics simultaneously
- [API Reference](../reference/api.md): full parameter documentation for `OptunaSearchCV`, `Sampler`, `Storage`, `Callback`
