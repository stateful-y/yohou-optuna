# How to Run a Multi-Metric Search

This guide shows you how to evaluate multiple scoring metrics simultaneously during hyperparameter search and select the best forecaster based on a chosen metric.

## Prerequisites

- Yohou-Optuna installed ([Getting Started](../tutorials/getting-started.md))
- A forecaster and search space set up

## Define Multiple Scorers

Pass a list of scorers to the `scoring` parameter. Each scorer is evaluated on every cross-validation fold:

```python
from yohou.metrics import MeanAbsoluteError, MeanSquaredError, MeanAbsolutePercentageError
from yohou_optuna import OptunaSearchCV

search = OptunaSearchCV(
    forecaster=forecaster,
    param_distributions=distributions,
    n_trials=30,
    scoring=[
        MeanAbsoluteError(),
        MeanSquaredError(),
        MeanAbsolutePercentageError(),
    ],
    refit="MeanAbsoluteError",
)

search.fit(y_train, forecasting_horizon=12)
```

When `scoring` is a list, `refit` must be a string matching the name of one scorer. That scorer's results determine `best_params_`, `best_score_`, and which forecaster is stored in `best_forecaster_`.

## Use a Dict for Custom Names

Pass a dict to assign custom keys to scorers. The keys become the column names in `cv_results_`:

```python
search = OptunaSearchCV(
    forecaster=forecaster,
    param_distributions=distributions,
    n_trials=30,
    scoring={
        "mae": MeanAbsoluteError(),
        "rmse": MeanSquaredError(square_root=True),
    },
    refit="mae",
)
```

## Inspect Multi-Metric Results

After fitting, `cv_results_` contains one column per metric per split. Use polars to inspect:

```python
import polars as pl

results = pl.DataFrame(search.cv_results_)

# Each metric has mean and std columns
print(results.select(["params", "mean_test_mae", "mean_test_rmse"]).sort("mean_test_mae"))
```

The `multimetric_` attribute is `True` when multiple scorers were used:

```python
print(search.multimetric_)   # True
print(search.scorer_)        # dict of scorers keyed by name
```

## Predict with the Best Forecaster

`predict()` always uses the forecaster selected by `refit`, regardless of which scorers were used:

```python
y_pred = search.predict(forecasting_horizon=12)
```

## See Also

- [Configure OptunaSearchCV](configure.md) - sampler, callbacks, CV options
- [API Reference](../reference/api.md) - `OptunaSearchCV` parameter documentation
- [Multi-Metric Search example](/examples/multi_metric_search/) - interactive notebook
