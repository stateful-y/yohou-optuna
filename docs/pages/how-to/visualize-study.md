# How to Visualize a Search Study

This guide shows you how to analyze and visualize the results of an `OptunaSearchCV` search using Optuna's built-in plotting tools and yohou's forecast diagnostics.

## Prerequisites

- Yohou-Optuna installed ([Getting Started](../tutorials/getting-started.md))
- A fitted `OptunaSearchCV` with at least 10-20 trials

## Access the Optuna Study

After `fit()`, the `study_` attribute holds the complete Optuna study object:

```python
study = search.study_

print(f"Number of trials: {len(study.trials)}")
print(f"Best value: {study.best_value:.4f}")
print(f"Best params: {study.best_params}")
```

## Plot Optimization History

The optimization history shows how the best score evolved over trials. A downward curve indicates the sampler is finding improvements over time:

```python
import optuna.visualization as vis

fig = vis.plot_optimization_history(study)
fig.show()
```

If the curve plateaus early, you may have enough trials. If it is still decreasing at the end, consider adding more trials.

## Plot Parameter Importances

Parameter importances estimate which parameters had the largest impact on the objective score. Use this to focus future searches:

```python
fig = vis.plot_param_importances(study)
fig.show()
```

Parameters with low importance are candidates for removal from the search space to reduce trial time.

## Plot Contour and Parallel Coordinates

Contour plots show how pairs of parameters interact. Parallel coordinate plots show the parameter combinations across all trials:

```python
# Interaction between two specific parameters
fig = vis.plot_contour(study, params=["regressor__alpha", "observation_horizon"])
fig.show()

# All parameter combinations
fig = vis.plot_parallel_coordinate(study)
fig.show()
```

## Combine with Yohou Diagnostics

After inspecting the search, use yohou's plotting functions to evaluate the best forecaster's predictions:

```python
from yohou.plotting import plot_forecast

y_pred = search.predict(forecasting_horizon=12)

fig = plot_forecast(y_train=y_train, y_pred=y_pred, y_test=y_test)
fig.show()
```

## See Also

- [Configure OptunaSearchCV](configure.md) - persist studies for longer analysis sessions
- [API Reference](../reference/api.md) - `study_` and `cv_results_` attributes
- [Search Visualization example](/examples/search_visualization/) - interactive notebook with combined Optuna and yohou plots
