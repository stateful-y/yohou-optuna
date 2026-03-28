# How to Configure OptunaSearchCV

This guide shows you how to configure `OptunaSearchCV` for common search scenarios: changing the sampler, adding callbacks, persisting studies, and customizing cross-validation.

## Prerequisites

- Yohou-Optuna installed ([Getting Started](../tutorials/getting-started.md))
- Familiarity with `OptunaSearchCV.fit()` basics

## Choose a Sampler

The `sampler` parameter controls the optimization strategy. The default - TPE (Tree-structured Parzen Estimator) - works well for most cases by building a probabilistic model of the objective function as trials accumulate.

```python
from yohou_optuna import OptunaSearchCV, Sampler

search = OptunaSearchCV(
    forecaster=forecaster,
    param_distributions=distributions,
    n_trials=50,
    sampler=Sampler("TPESampler", seed=42),     # default: TPE
)
```

For other strategies, swap the sampler name:

```python
# CMA-ES: effective for continuous spaces with correlated parameters
sampler=Sampler("CmaEsSampler", seed=42)

# Gaussian Process: best for very small budgets (< 20 trials)
sampler=Sampler("GPSampler")

# Random: useful as a baseline or for reproducible ablations
sampler=Sampler("RandomSampler", seed=42)
```

Pass `seed` for reproducible results. Always use the `Sampler` wrapper rather than a raw Optuna sampler object - raw Optuna objects are not compatible with `clone()`.

!!! tip
    Start with TPE. Switch to CMA-ES only when you have a large all-continuous search space and notice slow convergence.

## Add Callbacks

Callbacks run after each completed trial. Use them for early stopping, logging, or custom logic:

```python
from yohou_optuna import Callback

search = OptunaSearchCV(
    forecaster=forecaster,
    param_distributions=distributions,
    n_trials=200,
    callbacks=[
        Callback("MaxTrialsCallback", n_trials=50),
    ],
)
```

`MaxTrialsCallback` stops the study once the specified number of trials completes, regardless of the `n_trials` setting on `OptunaSearchCV`. This is useful when you want to set a generous upper bound on trials but stop early once you have enough results.

Always use the `Callback` wrapper - not a raw Optuna callback - for the same cloneability reasons as `Sampler`.

## Persist and Resume Studies

For long-running or distributed searches, save the study to a storage backend:

```python
from yohou_optuna import Storage

search = OptunaSearchCV(
    forecaster=forecaster,
    param_distributions=distributions,
    n_trials=50,
    storage=Storage("RDBStorage", url="sqlite:///my_study.db"),
    study_name="ridge_air_passengers",
)

search.fit(y_train, forecasting_horizon=12)
```

To add more trials later, pass the existing study back to `fit()`:

```python
search.n_trials = 50  # 50 additional trials
search.fit(y_train, forecasting_horizon=12, study=search.study_)
```

!!! tip
    Always set a `study_name` when using storage. Without it, Optuna generates a random name, making it harder to resume correctly.

## Use a Custom CV Splitter

By default, `OptunaSearchCV` uses a 5-fold expanding window. Pass any Yohou splitter to change this:

```python
from yohou.model_selection import SlidingWindowSplitter

search = OptunaSearchCV(
    forecaster=forecaster,
    param_distributions=distributions,
    n_trials=30,
    cv=SlidingWindowSplitter(n_splits=5, window_size=24),
)
```

Use `ExpandingWindowSplitter` (default) when you want the model to see all available history as training progresses. Use `SlidingWindowSplitter` when you want a fixed-size training window, which is useful for data with concept drift.

## Handle Fitting Errors

If a trial's parameter combination causes a fitting error, `OptunaSearchCV` raises by default. Set `error_score` to assign a fallback score instead:

```python
search = OptunaSearchCV(
    forecaster=forecaster,
    param_distributions=distributions,
    n_trials=50,
    error_score=float("nan"),  # or a numeric penalty
)
```

Use `error_score=float("nan")` when you expect occasional fitting failures (e.g., ill-conditioned matrices at extreme parameter values) and do not want to abort the entire search.

## Collect Training Scores

To compute scores on the training folds in addition to the validation folds, set `return_train_score=True`:

```python
search = OptunaSearchCV(
    forecaster=forecaster,
    param_distributions=distributions,
    n_trials=30,
    return_train_score=True,
)

search.fit(y_train, forecasting_horizon=12)

# cv_results_ now contains train_score columns alongside test_score columns
import polars as pl
results = pl.DataFrame(search.cv_results_)
print(results.select(["params", "mean_test_score", "mean_train_score"]))
```

Large gaps between training and test scores indicate overfitting to the training folds.

## See Also

- [About OptunaSearchCV](../explanation/concepts.md) - understand samplers, temporal CV, and wrapper classes
- [Multi-Metric Search](multi-metric-search.md) - evaluate multiple metrics simultaneously
- [API Reference](../reference/api.md) - full parameter documentation for `OptunaSearchCV`, `Sampler`, `Storage`, `Callback`
