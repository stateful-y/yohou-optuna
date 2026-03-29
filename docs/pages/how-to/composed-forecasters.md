# How to Tune Composed Forecasters

This guide shows you how to tune hyperparameters in composed forecasters (pipelines and combinations that nest multiple components) using `OptunaSearchCV`.

## Prerequisites

- Yohou-Optuna installed ([Getting Started](../tutorials/getting-started.md))
- Familiarity with Yohou's composition API (`DecompositionPipeline`, `ColumnForecaster`)

## Use Double-Underscore for Nested Parameters

Composed forecasters expose their components' parameters via the double-underscore convention. Given a `DecompositionPipeline` with a `deseason` step containing a `PointReductionForecaster`, the nested `alpha` parameter is accessed as `deseason__regressor__alpha`:

```python
from yohou.compose import DecompositionPipeline
from yohou.stationarity import LinearTrendForecaster, STLDecomposer
from yohou.point import PointReductionForecaster
from sklearn.linear_model import Ridge
from optuna.distributions import FloatDistribution, IntDistribution
from yohou_optuna import OptunaSearchCV

pipeline = DecompositionPipeline(
    decomposer=STLDecomposer(),
    detrend=LinearTrendForecaster(),
    deseason=PointReductionForecaster(regressor=Ridge()),
)

param_distributions = {
    "deseason__regressor__alpha": FloatDistribution(1e-4, 10.0, log=True),
    "deseason__observation_horizon": IntDistribution(3, 30),
    "decomposer__period": IntDistribution(4, 24),
}

search = OptunaSearchCV(
    forecaster=pipeline,
    param_distributions=param_distributions,
    n_trials=50,
)

search.fit(y_train, forecasting_horizon=12)
```

Call `pipeline.get_params(deep=True)` to discover the full set of tunable parameter names.

## Tune a ColumnForecaster

`ColumnForecaster` fits one forecaster per target column (multivariate targets). Tune parameters for each column's forecaster using the component name as prefix:

```python
from yohou.compose import ColumnForecaster

column_forecaster = ColumnForecaster(
    forecasters={
        "sales": PointReductionForecaster(regressor=Ridge()),
        "returns": PointReductionForecaster(regressor=Ridge()),
    }
)

param_distributions = {
    "sales__regressor__alpha": FloatDistribution(1e-4, 10.0, log=True),
    "returns__regressor__alpha": FloatDistribution(1e-4, 10.0, log=True),
}
```

## See Also

- [Collect Training Scores](configure.md#collect-training-scores) - compare training and validation scores to detect overfitting
- [Configure OptunaSearchCV](configure.md) - sampler and CV options
- [API Reference](../reference/api.md) - full parameter documentation
- [Composed Forecaster Tuning example](/examples/composed_tuning/) - interactive notebook
