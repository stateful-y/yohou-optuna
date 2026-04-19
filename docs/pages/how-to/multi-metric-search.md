# How to Run a Multi-Metric Search

This guide shows you how to evaluate multiple scoring metrics simultaneously during hyperparameter search and select the best forecaster based on a chosen metric.

!!! tip "Interactive notebook"
    See the companion notebook for a runnable example.
    [View](/examples/multi_metric_search/) · [Open in marimo](/examples/multi_metric_search/edit/)

## Prerequisites

- Yohou-Optuna installed ([Getting Started](../tutorials/getting-started.md))
- Familiarity with `OptunaSearchCV.fit()` basics

## Define Multiple Scorers

Pass a dictionary of scorers to the `scoring` parameter and set `refit` to the key that should drive model selection. Each scorer is evaluated on every cross-validation fold:

```python
import optuna
from optuna.distributions import CategoricalDistribution, FloatDistribution
from sklearn.linear_model import Ridge
from yohou.metrics import MeanAbsoluteError, MeanSquaredError, RootMeanSquaredError
from yohou.point import PointReductionForecaster
from yohou_optuna import OptunaSearchCV, Sampler

search = OptunaSearchCV(
    forecaster=PointReductionForecaster(estimator=Ridge()),
    param_distributions={
        "estimator__alpha": FloatDistribution(0.001, 100.0, log=True),
        "estimator__fit_intercept": CategoricalDistribution([True, False]),
    },
    scoring={
        "mae": MeanAbsoluteError(),
        "rmse": RootMeanSquaredError(),
        "mse": MeanSquaredError(),
    },
    sampler=Sampler(sampler=optuna.samplers.TPESampler, seed=42),
    n_trials=30,
    refit="mae",
    return_train_score=True,
)

search.fit(y_train, forecasting_horizon=12)
```

The `refit` key determines which metric selects the best forecaster. All metrics are still computed and stored for comparison.

## Use a List for Default Names

Pass a list of scorers instead of a dict. The class name becomes the column key in `cv_results_`:

```python
search = OptunaSearchCV(
    forecaster=PointReductionForecaster(estimator=Ridge()),
    param_distributions=distributions,
    scoring=[
        MeanAbsoluteError(),
        RootMeanSquaredError(),
    ],
    n_trials=30,
    refit="MeanAbsoluteError",
)
```

## Inspect Multi-Metric Results

After fitting, `cv_results_` contains mean scores and independent rankings for each metric. Use polars to compare:

```python
import polars as pl

results = pl.DataFrame(search.cv_results_)

# Each metric has mean, std, and rank columns
print(
    results.select(["params", "mean_test_mae", "mean_test_rmse", "rank_test_mae", "rank_test_rmse"])
    .sort("rank_test_mae")
)
```

Different metrics can rank trials differently. The `rank_test_*` columns let you see whether the best trial by MAE is also the best by RMSE. Setting `return_train_score=True` adds `mean_train_*` columns for overfitting diagnostics: large gaps between train and test scores suggest the model is memorizing the data.

## Predict with the Best Forecaster

`predict()` always uses the forecaster selected by `refit`, regardless of which scorers were used:

```python
y_pred = search.predict(forecasting_horizon=12)
```

## See Also

- [Configure OptunaSearchCV](configure.md): sampler, callbacks, CV options
- [API Reference](../reference/api.md): `OptunaSearchCV` parameter documentation
