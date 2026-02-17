# Getting Started

This guide will help you install and start using Yohou-Optuna in minutes.

## Installation

### Step 1: Install the package

Choose your preferred package manager:

=== "pip"

    ```bash
    pip install yohou_optuna
    ```

=== "uv"

    ```bash
    uv add yohou_optuna
    ```

### Step 2: Verify installation

```python
import yohou_optuna
print(yohou_optuna.__version__)
```

## Basic Usage

### Step 1: Import components

```python
import polars as pl
from sklearn.linear_model import Ridge

from yohou.point import PointReductionForecaster
from yohou_optuna import OptunaSearchCV
from optuna.distributions import FloatDistribution, IntDistribution
```

### Step 2: Define search space and fit

```python
# Create a forecaster
forecaster = PointReductionForecaster(regressor=Ridge())

# Define Optuna distributions
param_distributions = {
    "regressor__alpha": FloatDistribution(1e-4, 10.0, log=True),
    "observation_horizon": IntDistribution(3, 30),
}

# Create the searcher
search = OptunaSearchCV(
    forecaster=forecaster,
    param_distributions=param_distributions,
    n_trials=20,
)

# Fit on time series data (y must have a "time" column)
search.fit(y_train, X_train, forecasting_horizon=5)
```

### Step 3: Use results

```python
# Best forecaster is already fitted
print(search.best_params_)
print(search.best_score_)

# Predict directly — OptunaSearchCV is a forecaster
y_pred = search.predict(forecasting_horizon=5)

# Access the underlying Optuna study
print(search.study_)
```

## Complete Example

Here's a complete working example:

```python
import polars as pl
from sklearn.linear_model import Ridge

from yohou.datasets import load_air_passengers
from yohou.metrics import MeanAbsoluteError
from yohou.model_selection import ExpandingWindowSplitter
from yohou.point import PointReductionForecaster
from yohou_optuna import OptunaSearchCV
from optuna.distributions import FloatDistribution, IntDistribution

# Load data
y = load_air_passengers()

# Split into train/test
y_train = y.head(120)
y_test = y.tail(24)

# Set up search
forecaster = PointReductionForecaster(regressor=Ridge())
param_distributions = {
    "regressor__alpha": FloatDistribution(1e-4, 10.0, log=True),
    "observation_horizon": IntDistribution(3, 30),
}

search = OptunaSearchCV(
    forecaster=forecaster,
    param_distributions=param_distributions,
    n_trials=30,
    scoring=MeanAbsoluteError(),
    cv=ExpandingWindowSplitter(n_splits=3),
)

search.fit(y_train, forecasting_horizon=12)

# Results
print(f"Best score: {search.best_score_:.4f}")
print(f"Best params: {search.best_params_}")

# Predict
y_pred = search.predict(forecasting_horizon=12)
print(y_pred)
```

## Try Interactive Examples

For hands-on learning with interactive notebooks, see the [Examples](examples.md) page where you can:

- Run code directly in your browser
- Experiment with different parameters
- See visual outputs in real-time

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

Now that you have Yohou-Optuna installed and running:

- **Learn the concepts**: Read the [User Guide](user-guide.md) to understand core concepts and capabilities
- **Explore examples**: Check out the [Examples](examples.md) for real-world use cases
- **Dive into the API**: Browse the [API Reference](api-reference.md) for detailed documentation
