# Getting Started

In this tutorial, we will run our first Bayesian hyperparameter search for a time series forecaster. Along the way, we will install Yohou-Optuna, define a search space using Optuna distributions, fit an `OptunaSearchCV`, and predict with the best-found forecaster.

!!! tip "Interactive notebook"
    Follow along in the Quickstart notebook for a hands-on version of this tutorial.
    [View](/examples/optuna_search/) · [Open in marimo](/examples/optuna_search/edit/)

## Prerequisites

- Python 3.11+ installed
- A terminal or command prompt

## Step 1: Install Yohou-Optuna

=== "pip"

    ```bash
    pip install yohou_optuna
    ```

=== "uv"

    ```bash
    uv add yohou_optuna
    ```

Verify the installation works:

```python
import yohou_optuna
print(yohou_optuna.__version__)
```

The output should show a version string such as `0.1.0-alpha.2`.

## Step 2: Set Up a Forecaster

We will tune a `PointReductionForecaster`, a forecaster that converts a time series problem into a regression problem using lag features. `OptunaSearchCV` also works with interval forecasters (see the [API Reference](../reference/api.md) for details).

First, import what we need:

```python
import optuna
import polars as pl
from sklearn.linear_model import Ridge

from yohou.datasets import load_air_passengers
from yohou.metrics import MeanAbsoluteError
from yohou.model_selection import ExpandingWindowSplitter
from yohou.point import PointReductionForecaster
from yohou_optuna import OptunaSearchCV, Sampler
from optuna.distributions import CategoricalDistribution, FloatDistribution
```

Now let us load a dataset and split it into training and test sets:

```python
y = load_air_passengers()

y_train = y.head(120)
y_test = y.tail(24)
```

Notice that `y` is a polars DataFrame with a `time` column.

Create the base forecaster we want to tune:

```python
forecaster = PointReductionForecaster(estimator=Ridge())
```

## Step 3: Define the Search Space

We define the search space using Optuna distribution objects:

```python
param_distributions = {
    "estimator__alpha": FloatDistribution(0.001, 100.0, log=True),
    "estimator__fit_intercept": CategoricalDistribution([True, False]),
}
```

Notice that `estimator__alpha` uses a double-underscore to route `alpha` to the estimator inside the forecaster. See [About OptunaSearchCV](../explanation/concepts.md) for details on distributions and parameter routing.

## Step 4: Run the Search

Create an `OptunaSearchCV` and call `fit()`. This will run 30 trials, each one evaluating a different parameter combination via cross-validation:

```python
search = OptunaSearchCV(
    forecaster=forecaster,
    param_distributions=param_distributions,
    n_trials=20,
    scoring=MeanAbsoluteError(),
    cv=ExpandingWindowSplitter(n_splits=3),
    sampler=Sampler(sampler=optuna.samplers.TPESampler, seed=42),
)

search.fit(y_train, forecasting_horizon=12)
```

You should see Optuna logging trial results as they complete. After all trials finish, the best-found forecaster is automatically refit on the full training set.

## Step 5: Inspect the Results

Let us check what parameters were found:

```python
print(f"Best score: {search.best_score_:.4f}")
print(f"Best params: {search.best_params_}")
```

The output should look something like:

```text
Best score: 12.3456
Best params: {'estimator__alpha': 0.0137, 'estimator__fit_intercept': True}
```

The `study_` attribute gives us access to the full Optuna study, including all 30 trials with their scores and parameters:

```python
print(f"Number of trials: {len(search.trials_)}")
```

## Step 6: Predict

`OptunaSearchCV` is itself a forecaster. We can call `predict()` directly because it delegates to `best_forecaster_`:

```python
y_pred = search.predict(forecasting_horizon=12)
print(y_pred)
```

The prediction covers the next 12 time steps after the training period.

## What We Built

You have:

- Installed Yohou-Optuna
- Created a `PointReductionForecaster` with a `Ridge` estimator
- Defined a search space using `FloatDistribution` and `CategoricalDistribution`
- Fitted an `OptunaSearchCV` with a reproducible `Sampler` that ran 20 Bayesian trials with cross-validation
- Inspected the best score and parameters
- Predicted with the best-found forecaster

## Try Interactive Examples

For hands-on learning with interactive notebooks, see the [Examples](examples.md) page where you can run code directly in your browser or experiment with different parameters.

Or run locally:

=== "just"

    ```bash
    just example optuna_search.py
    ```

=== "uv run"

    ```bash
    uv run marimo edit examples/optuna_search.py
    ```

## Next Steps

- **Understand the design**: Read [About OptunaSearchCV](../explanation/concepts.md) to understand the object model, the search lifecycle, and wrapper classes
- **Explore more examples**: Browse the [Examples](examples.md) for interactive notebooks covering panel data, multi-metric search, and visualization
- **Configure the search**: See [Configure OptunaSearchCV](../how-to/configure.md) for sampler selection, callbacks, and study persistence
- **Browse the API**: See the [API Reference](../reference/api.md) for all parameters and attributes
