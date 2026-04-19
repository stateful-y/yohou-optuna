# How to Tune a Forecaster on Panel Data

This guide shows you how to use `OptunaSearchCV` with panel (grouped) time series data, where multiple related series share a common time index.

!!! tip "Interactive notebook"
    See the companion notebook for a runnable example.
    [View](/examples/panel_tuning/) · [Open in marimo](/examples/panel_tuning/edit/)

## Prerequisites

- Yohou-Optuna installed ([Getting Started](../tutorials/getting-started.md))
- Familiarity with Yohou's panel data format (a DataFrame with a `time` column and one or more group columns)

## Pass Panel Data Directly

`OptunaSearchCV` handles panel data transparently via `fit()`. The cross-validation splitter partitions rows by time index, preserving all groups in every fold:

```python
import optuna
from optuna.distributions import CategoricalDistribution, FloatDistribution
from sklearn.linear_model import Ridge
from yohou.datasets import load_australian_tourism
from yohou.metrics import MeanAbsoluteError
from yohou.model_selection import ExpandingWindowSplitter
from yohou.point import PointReductionForecaster
from yohou_optuna import OptunaSearchCV, Sampler

# Load panel dataset: multiple regional tourism series
y_panel = load_australian_tourism()
y_train = y_panel.head(60)
y_test = y_panel.tail(20)

forecaster = PointReductionForecaster(estimator=Ridge())

param_distributions = {
    "estimator__alpha": FloatDistribution(1e-4, 10.0, log=True),
    "estimator__fit_intercept": CategoricalDistribution([True, False]),
}

search = OptunaSearchCV(
    forecaster=forecaster,
    param_distributions=param_distributions,
    scoring=MeanAbsoluteError(),
    n_trials=30,
    cv=ExpandingWindowSplitter(n_splits=3, test_size=4),
    sampler=Sampler(sampler=optuna.samplers.TPESampler, seed=42),
)

search.fit(y_train, forecasting_horizon=4)
```

The scores across folds are averaged over all groups and time steps. The `panel_strategy` parameter on the forecaster controls how groups are handled: `"global"` (default) fits a single model on all groups, while `"multivariate"` treats each group as a separate target column.

## Compare Samplers on Panel Data

Panel data searches can be slower per trial due to the larger dataset size. If you need to benchmark samplers, swap the sampler and compare `best_score_`:

```python
random_search = OptunaSearchCV(
    forecaster=forecaster,
    param_distributions=param_distributions,
    scoring=MeanAbsoluteError(),
    n_trials=30,
    cv=ExpandingWindowSplitter(n_splits=3, test_size=4),
    sampler=Sampler(sampler=optuna.samplers.RandomSampler, seed=42),
)
random_search.fit(y_train, forecasting_horizon=4)

print(f"Random best: {random_search.best_score_:.4f}")
print(f"TPE best:    {search.best_score_:.4f}")
```

## Predict and Visualize Panel Forecasts

The fitted `OptunaSearchCV` predicts for all groups simultaneously. Use `plot_forecast` to compare against held-out data:

```python
from yohou.plotting import plot_forecast

y_pred = search.predict(forecasting_horizon=4)

plot_forecast(y_test, y_pred, y_train=y_train, n_history=12)
```

The output DataFrame retains the same group structure as the input `y_train`.

## See Also

- [Configure OptunaSearchCV](configure.md): sampler options, callbacks, and early stopping
- [Composed Forecasters](composed-forecasters.md): tune nested pipelines on panel data
- [API Reference](../reference/api.md): full parameter documentation
