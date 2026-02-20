<p align="center">
  <picture>
    <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/stateful-y/yohou-optuna/main/docs/assets/logo_light.png">
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/stateful-y/yohou-optuna/main/docs/assets/logo_dark.png">
    <img src="https://raw.githubusercontent.com/stateful-y/yohou-optuna/main/docs/assets/logo_light.png" alt="Yohou-Optuna">
  </picture>
</p>


[![Python Version](https://img.shields.io/pypi/pyversions/yohou_optuna)](https://pypi.org/project/yohou_optuna/)
[![License](https://img.shields.io/github/license/stateful-y/yohou-optuna)](https://github.com/stateful-y/yohou-optuna/blob/main/LICENSE)
[![PyPI Version](https://img.shields.io/pypi/v/yohou_optuna)](https://pypi.org/project/yohou_optuna/)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/yohou_optuna)](https://anaconda.org/conda-forge/yohou_optuna)
[![codecov](https://codecov.io/gh/stateful-y/yohou-optuna/branch/main/graph/badge.svg)](https://codecov.io/gh/stateful-y/yohou-optuna)

## What is Yohou-Optuna?

**Yohou-Optuna** brings [Optuna](https://optuna.org/)'s hyperparameter optimization to [Yohou](https://github.com/stateful-y/yohou), providing a Yohou-compatible search class for time series forecasting.

This integration replaces grid and random search with adaptive sampling (TPE, CMA-ES, and more) while keeping Yohou's forecasting API intact. After fitting, `OptunaSearchCV` behaves like a Yohou forecaster, so you can call `predict`, `observe`, and `observe_predict` directly.

It integrates with Optuna's distributions, samplers, and storages, and wraps them for sklearn-style cloning and serialization.

## What are the features of Yohou-Optuna?

- **Adaptive optimization**: Run Optuna studies over Yohou forecasters with TPE, CMA-ES, and other samplers to find better configurations in fewer trials.
- **Forecaster-native API**: `OptunaSearchCV` is a forecaster after fitting, so you can call `predict`, `observe`, `observe_predict`, and interval methods.
- **Clone-safe wrappers**: `Sampler`, `Storage`, and `Callback` wrappers ensure Optuna objects survive sklearn cloning and serialization.
- **Time-series CV support**: Works with Yohou splitters for proper temporal validation and scorer integration.
- **Multi-metric evaluation**: Evaluate multiple scorers and refit on the one that matters most for your use case.
- **(Experimental) Persistence workflows**: Resume studies with storage-backed optimization and continue tuning over time.

## How to install Yohou-Optuna?

Install the Yohou-Optuna package using `pip`:

```bash
pip install yohou_optuna
```

or using `uv`:

```bash
uv pip install yohou_optuna
```

or using `conda`:

```bash
conda install -c conda-forge yohou_optuna
```

or using `mamba`:

```bash
mamba install -c conda-forge yohou_optuna
```

or alternatively, add `yohou_optuna` to your `requirements.txt` or `pyproject.toml` file.

## How to get started with Yohou-Optuna?

### 1. Prepare a forecaster and search space

Define a Yohou forecaster and Optuna distributions for the parameters you want to tune.

```python
from sklearn.linear_model import Ridge
from optuna.distributions import FloatDistribution, IntDistribution

from yohou.point import PointReductionForecaster
from yohou_optuna import OptunaSearchCV

forecaster = PointReductionForecaster(estimator=Ridge())
param_distributions = {
    "estimator__alpha": FloatDistribution(1e-4, 10.0, log=True),
    "observation_horizon": IntDistribution(3, 30),
}

search = OptunaSearchCV(
    forecaster=forecaster,
    param_distributions=param_distributions,
    n_trials=30,
)
```

### 2. Fit the searcher

Fit the searcher on your time series data (polars DataFrame with a `time` column).

```python
search.fit(y_train, X_train, forecasting_horizon=12)
```

### 3. Predict with the best forecaster

After fitting, `search` behaves like a Yohou forecaster.

```python
y_pred = search.predict(forecasting_horizon=12)
print(search.best_params_)
```

## How do I use Yohou-Optuna?

Full documentation is available at [https://yohou-optuna.readthedocs.io/](https://yohou-optuna.readthedocs.io/).

Interactive examples are available in the `examples/` directory:

- **Online**: [https://yohou-optuna.readthedocs.io/en/latest/pages/examples/](https://yohou-optuna.readthedocs.io/en/latest/pages/examples/)
- **Locally**: Run `marimo edit examples/optuna_search.py` to open an interactive notebook

## Can I contribute?

We welcome contributions, feedback, and questions:

- **Report issues or request features**: [GitHub Issues](https://github.com/stateful-y/yohou-optuna/issues)
- **Join the discussion**: [GitHub Discussions](https://github.com/stateful-y/yohou-optuna/discussions)
- **Contributing Guide**: [CONTRIBUTING.md](https://github.com/stateful-y/yohou-optuna/blob/main/CONTRIBUTING.md)

If you are interested in becoming a maintainer or taking a more active role, please reach out to Guillaume Tauzin on [GitHub Discussions](https://github.com/stateful-y/yohou-optuna/discussions).

## Where can I learn more?

Here are the main Yohou-Optuna resources:

- Full documentation: [https://yohou-optuna.readthedocs.io/](https://yohou-optuna.readthedocs.io/)
- GitHub Discussions: [https://github.com/stateful-y/yohou-optuna/discussions](https://github.com/stateful-y/yohou-optuna/discussions)
- Interactive Examples: [https://yohou-optuna.readthedocs.io/en/latest/pages/examples/](https://yohou-optuna.readthedocs.io/en/latest/pages/examples/)

For questions and discussions, you can also open a [discussion](https://github.com/stateful-y/yohou-optuna/discussions).

## License

This project is licensed under the terms of the [Apache-2.0 License](https://github.com/stateful-y/yohou-optuna/blob/main/LICENSE).

<p align="center">
  <a href="https://stateful-y.io">
    <img src="docs/assets/made_by_stateful-y.png" alt="Made by stateful-y" width="200">
  </a>
</p>
