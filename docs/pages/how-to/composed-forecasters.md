# How to Tune Composed Forecasters

This guide shows you how to tune hyperparameters in composed forecasters (pipelines and combinations that nest multiple components) using `OptunaSearchCV`.

!!! tip "Interactive notebook"
    See the companion notebook for a runnable example.
    [View](/examples/composed_tuning/) · [Open in marimo](/examples/composed_tuning/edit/)

## Prerequisites

- Yohou-Optuna installed ([Getting Started](../tutorials/getting-started.md))
- Familiarity with Yohou's composition API (`DecompositionPipeline`, `ColumnForecaster`)

## Use Double-Underscore for Nested Parameters

Composed forecasters expose their components' parameters via the double-underscore convention. A `PointReductionForecaster` with an `estimator` and a `feature_transformer` exposes nested parameters like `estimator__alpha` and `feature_transformer__lag`:

```python
import optuna
from optuna.distributions import FloatDistribution, IntDistribution
from sklearn.linear_model import Ridge

from yohou.metrics import MeanAbsoluteError
from yohou.model_selection import ExpandingWindowSplitter
from yohou.point import PointReductionForecaster
from yohou.preprocessing import LagTransformer
from yohou_optuna import OptunaSearchCV, Sampler

forecaster = PointReductionForecaster(
    estimator=Ridge(),
    feature_transformer=LagTransformer(lag=6),
)

param_distributions = {
    "estimator__alpha": FloatDistribution(0.001, 100.0, log=True),
    "feature_transformer__lag": IntDistribution(3, 24),
}

search = OptunaSearchCV(
    forecaster=forecaster,
    param_distributions=param_distributions,
    scoring=MeanAbsoluteError(),
    sampler=Sampler(sampler=optuna.samplers.TPESampler, seed=42),
    n_trials=20,
    cv=ExpandingWindowSplitter(n_splits=3, test_size=24),
)

search.fit(y_train, forecasting_horizon=24)
```

Call `forecaster.get_params(deep=True)` to discover the full set of tunable parameter names.

## Tune a DecompositionPipeline

`DecompositionPipeline` chains multiple forecasters that each remove a component (trend, seasonality) from the series. It takes a list of `(name, forecaster)` tuples. Nested parameters use the step name as prefix:

```python
from yohou.compose import DecompositionPipeline
from yohou.stationarity import PolynomialTrendForecaster, FourierSeasonalityForecaster

pipeline = DecompositionPipeline(
    forecasters=[
        ("trend", PolynomialTrendForecaster()),
        ("season", FourierSeasonalityForecaster(seasonality=12)),
    ]
)

param_distributions = {
    "trend__degree": IntDistribution(1, 3),
    "season__seasonality": IntDistribution(6, 24),
    "season__harmonics": IntDistribution(1, 6),
}
```

Call `pipeline.get_params(deep=True)` to see all available parameter paths.

## Tune a ColumnForecaster

`ColumnForecaster` fits one forecaster per target column (multivariate targets). It takes a list of `(name, forecaster, columns)` tuples. Tune parameters for each column's forecaster using the component name as prefix:

```python
from yohou.compose import ColumnForecaster

column_forecaster = ColumnForecaster(
    forecasters=[
        ("sales", PointReductionForecaster(estimator=Ridge()), "sales"),
        ("returns", PointReductionForecaster(estimator=Ridge()), "returns"),
    ]
)

param_distributions = {
    "sales__estimator__alpha": FloatDistribution(1e-4, 10.0, log=True),
    "returns__estimator__alpha": FloatDistribution(1e-4, 10.0, log=True),
}
```

## See Also

- [Collect Training Scores](configure.md#collect-training-scores): compare training and validation scores to detect overfitting
- [Configure OptunaSearchCV](configure.md): sampler and CV options
- [API Reference](../reference/api.md): full parameter documentation
