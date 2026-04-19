# How to Visualize a Search Study

This guide shows you how to analyze and visualize the results of an `OptunaSearchCV` search using Optuna's built-in plotting tools and yohou's forecast diagnostics.

!!! tip "Interactive notebook"
    See the companion notebook for a runnable example.
    [View](/examples/search_visualization/) · [Open in marimo](/examples/search_visualization/edit/)

## Prerequisites

- Yohou-Optuna installed ([Getting Started](../tutorials/getting-started.md))
- A fitted `OptunaSearchCV` with at least 10-20 trials

## Set Up a Fitted Search

The examples below assume a fitted search. Here is a minimal setup:

```python
import optuna
import optuna.visualization as vis
from optuna.distributions import CategoricalDistribution, FloatDistribution
from sklearn.linear_model import Ridge

from yohou.datasets import load_vic_electricity
from yohou.metrics import MeanAbsoluteError
from yohou.plotting import plot_cv_results_scatter, plot_forecast, plot_residual_time_series
from yohou.point import PointReductionForecaster
from yohou_optuna import OptunaSearchCV, Sampler

y = load_vic_electricity().select(["time", "Demand"]).head(500)
y_train = y.head(400)
y_test = y.tail(24)

search = OptunaSearchCV(
    forecaster=PointReductionForecaster(estimator=Ridge()),
    param_distributions={
        "estimator__alpha": FloatDistribution(0.001, 100.0, log=True),
        "estimator__fit_intercept": CategoricalDistribution([True, False]),
    },
    scoring=MeanAbsoluteError(),
    sampler=Sampler(sampler=optuna.samplers.TPESampler, seed=42),
    n_trials=25,
    cv=3,
)

search.fit(y_train, forecasting_horizon=24)
```

## Access the Optuna Study

After `fit()`, the `study_` attribute holds the complete Optuna study object:

```python
study = search.study_

print(f"Number of trials: {len(study.trials)}")
print(f"Best value: {study.best_value:.4f}")
print(f"Best params: {study.best_params}")
```

## Plot Optimization History

The optimization history shows how the best score evolved over trials:

```python
fig = vis.plot_optimization_history(study)
fig.show()
```

## Plot Parameter Importances

Parameter importances estimate which parameters had the largest impact on the objective score:

```python
fig = vis.plot_param_importances(study)
fig.show()
```

## Plot Slice and Contour

Slice plots show how the objective responds to each parameter individually. Contour plots show how pairs of parameters interact:

```python
# Per-parameter objective values across trials
fig = vis.plot_slice(study)
fig.show()

# Interaction between two specific parameters
fig = vis.plot_contour(study, params=["estimator__alpha", "estimator__fit_intercept"])
fig.show()
```

## Plot Parallel Coordinates

Parallel coordinate plots show the parameter combinations across all trials:

```python
fig = vis.plot_parallel_coordinate(study)
fig.show()
```

## Combine with Yohou Diagnostics

Yohou provides plotting functions that complement Optuna's study-level visualizations with forecast-level analysis.

### CV Results Scatter

`plot_cv_results_scatter` shows how cross-validation scores relate to a specific parameter:

```python
fig = plot_cv_results_scatter(
    search.cv_results_,
    param_name="estimator__alpha",
    higher_is_better=False,
    title="CV Score vs Regularization Strength",
)
fig.show()
```

### Forecast vs Actual

Use `plot_forecast` to compare the best forecaster's predictions against the held-out test set:

```python
y_pred = search.predict(forecasting_horizon=24)

fig = plot_forecast(y_test, y_pred, y_train=y_train, n_history=48)
fig.show()
```

### Residual Analysis

Use `plot_residual_time_series` to inspect prediction errors over time:

```python
fig = plot_residual_time_series(y_pred, y_test, title="Forecast Residuals")
fig.show()
```

## See Also

- [Configure OptunaSearchCV](configure.md): persist studies for longer analysis sessions
- [API Reference](../reference/api.md): `study_` and `cv_results_` attributes
