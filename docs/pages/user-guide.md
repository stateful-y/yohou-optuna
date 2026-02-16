# User Guide

This guide provides comprehensive documentation for Yohou-Optuna.

## Overview

Yohou-Optuna brings Optuna's Bayesian hyperparameter optimization to Yohou's time series forecasting framework. Unlike `GridSearchCV` or `RandomizedSearchCV`, which evaluate a fixed or random set of candidates, `OptunaSearchCV` uses Optuna's samplers (TPE, CMA-ES, GP, etc.) to adaptively explore the search space, focusing trials on promising regions.

`OptunaSearchCV` extends Yohou's `BaseSearchCV`, which itself extends `BaseForecaster`. This means `OptunaSearchCV` **is** a forecaster: after fitting it behaves exactly like its best-found forecaster, supporting `predict()`, `observe()`, `observe_predict()`, and all other yohou forecasting methods.

The wrapper classes `Sampler`, `Storage`, and `Callback` let you pass Optuna objects through sklearn's `clone()` and serialization boundaries, which is essential for cross-validation and persistence.

## Prerequisites

Before diving into Yohou-Optuna, it's helpful to understand:

### Yohou's Forecasting API

Yohou forecasters use `fit(y, X, forecasting_horizon)` (not sklearn's `fit(X, y)`). The target `y` is a polars DataFrame with a mandatory "time" column. Exogenous features `X` are optional.

Learn more: [Yohou Documentation](https://stateful-y.github.io/yohou/)

### Optuna Distributions

Optuna defines parameter search spaces using distribution objects: `FloatDistribution`, `IntDistribution`, and `CategoricalDistribution`. These support log-scaling, step sizes, and bounded ranges.

Learn more: [Optuna Distributions](https://optuna.readthedocs.io/en/stable/reference/distributions.html)

## Why Yohou-Optuna?

Yohou ships with `GridSearchCV` and `RandomizedSearchCV` for hyperparameter search. These work well for small search spaces, but become inefficient as the number of parameters or their ranges grow. Optuna's samplers adaptively allocate trials to promising regions, typically reaching better solutions in fewer evaluations.

### For Yohou Users

If you already use `GridSearchCV` or `RandomizedSearchCV`, switching to `OptunaSearchCV` is straightforward:

- **Same API**: Replace `GridSearchCV` with `OptunaSearchCV` and `param_grid` with `param_distributions`. Everything else (`fit()`, `predict()`, `best_params_`, `cv_results_`) stays the same.
- **Smarter search**: TPE (Tree-structured Parzen Estimator) focuses on promising parameter regions instead of exhaustive enumeration.
- **Persistence**: Save and resume studies with `Storage`, enabling long-running or distributed searches.
- **Callbacks**: Monitor progress, implement early stopping, or log to external systems using `Callback` wrappers.

### For Optuna Users

If you're familiar with Optuna but new to Yohou, you'll appreciate:

- **Forecaster compatibility**: `OptunaSearchCV` wraps any yohou forecaster and handles temporal cross-validation automatically.
- **Panel data support**: Tune forecasters across multiple time series (panel data) without additional boilerplate.
- **Full forecaster API**: The fitted `OptunaSearchCV` object supports `predict()`, `observe()`, `observe_predict()`, and interval forecasting if the underlying forecaster supports it.

### Compared to Manual Optuna Studies

Writing a custom Optuna objective for time series is error-prone: you need to handle temporal splits, recursive prediction, proper scoring, and parameter routing. `OptunaSearchCV` handles all of this:

- **Automatic CV**: Uses yohou's splitters (`ExpandingWindowSplitter`, `SlidingWindowSplitter`) for proper temporal cross-validation.
- **Parameter routing**: Nested parameters like `regressor__alpha` are routed correctly through composition chains.
- **Multi-metric support**: Evaluate multiple metrics simultaneously with `scoring` as a list or dict.

## Core Concepts

### OptunaSearchCV Lifecycle

The search lifecycle follows yohou's standard forecaster pattern:

1. **Construction**: Define the forecaster, parameter distributions, number of trials, scoring, and CV strategy.
2. **`fit(y, X, forecasting_horizon)`**: Creates an Optuna study, runs `n_trials` trials, each evaluating a parameter combination via cross-validation. Refits the best forecaster on the full training data.
3. **`predict(forecasting_horizon)`**: Delegates to the best-found forecaster.
4. **`observe(y, X)`**: Observes new data for the best forecaster.

```python
from yohou_optuna import OptunaSearchCV

search = OptunaSearchCV(
    forecaster=forecaster,
    param_distributions=distributions,
    n_trials=50,
    scoring=scorer,
    cv=splitter,
)

# fit → creates study → runs trials → refits best
search.fit(y_train, X_train, forecasting_horizon=12)

# predict → delegates to best_forecaster_
y_pred = search.predict(forecasting_horizon=12)

# observe → feeds new data to best_forecaster_
search.observe(y_new, X_new)
y_pred_updated = search.predict(forecasting_horizon=12)
```

!!! example "Interactive Example"
    See [**Quickstart Search**](/examples/optuna_search/) ([View](/examples/optuna_search/) | [Editable](/examples/optuna_search/edit/)) for a complete walkthrough of the search lifecycle using the Air Passengers dataset.

### Wrapper Classes

Optuna's `BaseSampler`, `BaseStorage`, and callback objects are not compatible with sklearn's `clone()` (used internally during cross-validation). Yohou-Optuna provides wrapper classes that serialize the class name and constructor arguments, enabling safe cloning:

```python
from yohou_optuna import Sampler, Storage, Callback

# Wrap Optuna objects for clone()-safety
sampler = Sampler("TPESampler", seed=42)
storage = Storage("RDBStorage", url="sqlite:///study.db")
callback = Callback("MaxTrialsCallback", n_trials=100)
```

These wrappers are re-exported from [sklearn-optuna](https://github.com/stateful-y/sklearn-optuna) and instantiate the underlying Optuna objects lazily.

### Parameter Distributions

Use Optuna's native distribution objects to define search spaces:

```python
from optuna.distributions import (
    CategoricalDistribution,
    FloatDistribution,
    IntDistribution,
)

param_distributions = {
    # Log-scaled float (e.g., regularization strength)
    "regressor__alpha": FloatDistribution(1e-4, 10.0, log=True),

    # Integer range (e.g., window size)
    "observation_horizon": IntDistribution(3, 30),

    # Categorical choice (e.g., kernel type)
    "regressor__kernel": CategoricalDistribution(["linear", "rbf", "poly"]),

    # Float with step size
    "learning_rate": FloatDistribution(0.01, 1.0, step=0.01),
}
```

!!! example "Interactive Example"
    See [**Composed Forecaster Tuning**](/examples/composed_tuning/) ([View](/examples/composed_tuning/) | [Editable](/examples/composed_tuning/edit/)) for a demonstration of tuning nested parameters with `IntDistribution` and `FloatDistribution`.

## Key Features

### 1. Sampler Selection

Control the optimization strategy via the `sampler` parameter:

```python
search = OptunaSearchCV(
    forecaster=forecaster,
    param_distributions=distributions,
    n_trials=50,
    sampler=Sampler("TPESampler", seed=42),        # Default: TPE
    # sampler=Sampler("CmaEsSampler", seed=42),    # CMA-ES
    # sampler=Sampler("GPSampler"),                 # Gaussian Process
    # sampler=Sampler("RandomSampler", seed=42),    # Random (baseline)
)
```

!!! example "Interactive Example"
    See [**Panel Data Tuning**](/examples/panel_tuning/) ([View](/examples/panel_tuning/) | [Editable](/examples/panel_tuning/edit/)) for a side-by-side comparison of Random vs TPE samplers on the Australian Tourism dataset.

### 2. Callbacks

Use callbacks for early stopping, logging, or custom logic:

```python
from optuna.exceptions import TrialPruned

search = OptunaSearchCV(
    forecaster=forecaster,
    param_distributions=distributions,
    n_trials=100,
    callbacks=[
        Callback("MaxTrialsCallback", n_trials=100),
    ],
)
```

!!! example "Interactive Example"
    See [**Panel Data Tuning**](/examples/panel_tuning/) ([View](/examples/panel_tuning/) | [Editable](/examples/panel_tuning/edit/)) for a demonstration of `MaxTrialsCallback` for early stopping.

### 3. Study Persistence

Save and resume optimization studies using storage backends:

```python
search = OptunaSearchCV(
    forecaster=forecaster,
    param_distributions=distributions,
    n_trials=50,
    storage=Storage("RDBStorage", url="sqlite:///my_study.db"),
    study_name="forecaster_tuning",
)

# First run
search.fit(y_train, forecasting_horizon=12)

# Resume later (adds more trials to the same study)
search.n_trials = 50  # 50 more trials
search.fit(y_train, forecasting_horizon=12, study=search.study_)
```

### 4. Multi-Metric Evaluation

Evaluate multiple metrics simultaneously:

```python
from yohou.metrics import MeanAbsoluteError, MeanSquaredError

search = OptunaSearchCV(
    forecaster=forecaster,
    param_distributions=distributions,
    n_trials=30,
    scoring=[MeanAbsoluteError(), MeanSquaredError()],
    refit="MeanAbsoluteError",  # Which metric to use for selecting best
)
```

When `scoring` is a list or dict, `cv_results_` contains columns for each metric. Set `refit` to the metric name used for selecting the best forecaster.

!!! example "Interactive Example"
    See [**Multi-Metric Search**](/examples/multi_metric_search/) ([View](/examples/multi_metric_search/) | [Editable](/examples/multi_metric_search/edit/)) for evaluating MAE, RMSE, and MSE simultaneously on the ETT-M1 dataset.

### 5. Training Scores

Optionally compute scores on the training folds:

```python
search = OptunaSearchCV(
    forecaster=forecaster,
    param_distributions=distributions,
    n_trials=30,
    return_train_score=True,
)
```

### 6. Composed Forecasters

Tune nested parameters in composed forecasters like `DecompositionPipeline` or `ColumnForecaster`:

```python
from yohou.compose import DecompositionPipeline
from yohou.stationarity import LinearTrendForecaster, STLDecomposer

pipeline = DecompositionPipeline(
    decomposer=STLDecomposer(),
    detrend=LinearTrendForecaster(),
    deseason=PointReductionForecaster(regressor=Ridge()),
)

# Use __ separator for nested parameters
param_distributions = {
    "deseason__regressor__alpha": FloatDistribution(1e-4, 10.0, log=True),
    "deseason__observation_horizon": IntDistribution(3, 30),
    "decomposer__period": IntDistribution(4, 24),
}
```

!!! example "Interactive Example"
    See [**Composed Forecaster Tuning**](/examples/composed_tuning/) ([View](/examples/composed_tuning/) | [Editable](/examples/composed_tuning/edit/)) for tuning a `PointReductionForecaster` with a `LagTransformer` feature pipeline.

## Configuration

### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `forecaster` | BaseForecaster | *required* | The yohou forecaster to tune |
| `param_distributions` | dict | *required* | Optuna distribution objects keyed by parameter name |
| `n_trials` | int | `10` | Number of Optuna trials to run |
| `scoring` | scorer or list | `None` | Scorer(s) for evaluation (default: forecaster's default scorer) |
| `cv` | splitter or int | `5` | Cross-validation splitter or number of folds |
| `refit` | bool or str | `True` | Whether to refit the best forecaster; or metric name for multi-metric |
| `return_train_score` | bool | `False` | Whether to compute training fold scores |
| `sampler` | Sampler | `None` | Optuna sampler wrapper (default: TPE) |
| `storage` | Storage | `None` | Optuna storage wrapper for persistence |
| `callbacks` | list[Callback] | `None` | Optuna callback wrappers |
| `study_name` | str | `None` | Name for the Optuna study |
| `error_score` | float or "raise" | `"raise"` | Score to assign if fitting fails |

### Fitted Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `best_forecaster_` | BaseForecaster | The best-found forecaster, fitted on full training data |
| `best_params_` | dict | Parameters of the best trial |
| `best_score_` | float | Score of the best trial |
| `best_index_` | int | Index of the best trial in `cv_results_` |
| `cv_results_` | dict | Detailed results for all trials (scores, params, timings) |
| `study_` | optuna.Study | The underlying Optuna study object |
| `trials_` | list | List of completed Optuna trials |
| `scorer_` | scorer | The scorer used for evaluation |
| `multimetric_` | bool | Whether multiple metrics were used |
| `n_splits_` | int | Number of CV splits |

## Best Practices

### 1. Start with TPE

TPE (Tree-structured Parzen Estimator) is Optuna's default sampler and works well for most time series hyperparameter tuning tasks. Start with TPE before experimenting with CMA-ES or GP samplers.

### 2. Use Log-Scaling for Regularization

Parameters like regularization strength (`alpha`, `C`, `lambda`) typically span several orders of magnitude. Use `log=True` in `FloatDistribution` to search efficiently:

```python
"regressor__alpha": FloatDistribution(1e-4, 10.0, log=True)
```

### 3. Set Seeds for Reproducibility

Pass `seed` to the `Sampler` wrapper to get reproducible trial sequences:

```python
sampler = Sampler("TPESampler", seed=42)
```

## Limitations and Considerations

1. **No parallel trial execution**: Trials run sequentially within a single `fit()` call. For parallel execution, use Optuna's distributed optimization with a shared storage backend.

2. **Pruning not supported**: Optuna's trial pruning (early stopping of unpromising trials) is not yet integrated. All trials run to completion.

3. **Study resumption requires explicit `study` parameter**: To add trials to an existing study, pass the `study` object to `fit()`.

## FAQ

### How does OptunaSearchCV differ from GridSearchCV?

`GridSearchCV` evaluates every combination in a parameter grid. `OptunaSearchCV` uses Bayesian optimization to focus on promising regions, typically finding better solutions in fewer trials. Use `GridSearchCV` for small, discrete search spaces; use `OptunaSearchCV` for large or continuous spaces.

### Can I use OptunaSearchCV with interval forecasters?

Yes. If the underlying forecaster supports interval prediction, `OptunaSearchCV` inherits that capability. After fitting, call `predict_interval(forecasting_horizon, coverage_rates)` as usual.

### How do I access the Optuna study for visualization?

After fitting, use `search.study_` to access the underlying `optuna.Study` object. You can then use Optuna's built-in visualization tools:

```python
import optuna

optuna.visualization.plot_optimization_history(search.study_)
optuna.visualization.plot_param_importances(search.study_)
```

!!! example "Interactive Example"
    See [**Search Visualization**](/examples/search_visualization/) ([View](/examples/search_visualization/) | [Editable](/examples/search_visualization/edit/)) for a complete comparison of Optuna's optimization plots alongside yohou's forecast diagnostics.

## Next Steps

Now that you understand the core concepts and features:

- Follow the [Getting Started](getting-started.md) guide to start using Yohou-Optuna
- Explore the [Examples](examples.md) for interactive notebooks:
    - [**Quickstart Search**](/examples/optuna_search/) ([View](/examples/optuna_search/) | [Editable](/examples/optuna_search/edit/)): Your first hyperparameter search
    - [**Composed Tuning**](/examples/composed_tuning/) ([View](/examples/composed_tuning/) | [Editable](/examples/composed_tuning/edit/)): Nested parameter tuning
    - [**Multi-Metric Search**](/examples/multi_metric_search/) ([View](/examples/multi_metric_search/) | [Editable](/examples/multi_metric_search/edit/)): Multiple scoring metrics
    - [**Search Visualization**](/examples/search_visualization/) ([View](/examples/search_visualization/) | [Editable](/examples/search_visualization/edit/)): Optuna + yohou plots
    - [**Panel Data Tuning**](/examples/panel_tuning/) ([View](/examples/panel_tuning/) | [Editable](/examples/panel_tuning/edit/)): Grouped time series
- Check the [API Reference](api-reference.md) for detailed API documentation
