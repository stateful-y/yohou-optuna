# How to Tune a Forecaster on Panel Data

This guide shows you how to use `OptunaSearchCV` with panel (grouped) time series data - multiple related series sharing a common time index.

## Prerequisites

- Yohou-Optuna installed ([Getting Started](../tutorials/getting-started.md))
- Familiarity with Yohou's panel data format (a DataFrame with a `time` column and one or more group columns)

## Pass Panel Data Directly

`OptunaSearchCV` handles panel data transparently via `fit()`. The cross-validation splitter partitions rows by time index, preserving all groups in every fold:

```python
from yohou.datasets import load_australian_tourism
from yohou.model_selection import ExpandingWindowSplitter
from yohou.point import PointReductionForecaster
from yohou_optuna import OptunaSearchCV, Sampler
from optuna.distributions import CategoricalDistribution, FloatDistribution
from sklearn.linear_model import Ridge

# Load panel dataset: multiple regional tourism series
y_panel = load_australian_tourism()

forecaster = PointReductionForecaster(regressor=Ridge())

param_distributions = {
    "regressor__alpha": FloatDistribution(1e-4, 10.0, log=True),
    "regressor__fit_intercept": CategoricalDistribution([True, False]),
}

search = OptunaSearchCV(
    forecaster=forecaster,
    param_distributions=param_distributions,
    n_trials=30,
    cv=ExpandingWindowSplitter(n_splits=3),
    sampler=Sampler("TPESampler", seed=42),
)

search.fit(y_panel, forecasting_horizon=4)
```

The scores across folds are averaged over all groups and time steps.

## Compare Samplers on Panel Data

Panel data searches can be slower per trial due to the larger dataset size. If you need to benchmark samplers, run both and compare `best_score_`:

```python
from yohou_optuna import Sampler

# Random sampler as baseline
random_search = OptunaSearchCV(
    forecaster=forecaster,
    param_distributions=param_distributions,
    n_trials=30,
    sampler=Sampler("RandomSampler", seed=42),
)
random_search.fit(y_panel, forecasting_horizon=4)

# TPE sampler
tpe_search = OptunaSearchCV(
    forecaster=forecaster,
    param_distributions=param_distributions,
    n_trials=30,
    sampler=Sampler("TPESampler", seed=42),
)
tpe_search.fit(y_panel, forecasting_horizon=4)

print(f"Random best: {random_search.best_score_:.4f}")
print(f"TPE best:    {tpe_search.best_score_:.4f}")
```

## Predict Panel Forecasts

The fitted `OptunaSearchCV` predicts for all groups simultaneously:

```python
y_pred = search.predict(forecasting_horizon=4)
print(y_pred)
```

The output DataFrame retains the same group structure as the input `y_panel`.

## See Also

- [Configure OptunaSearchCV](configure.md) - sampler options and callbacks
- [API Reference](../reference/api.md) - full parameter documentation
- [Panel Data Tuning example](/examples/panel_tuning/) - interactive notebook with Australian Tourism dataset
