![](assets/logo_dark.png#only-dark){width=800}
![](assets/logo_light.png#only-light){width=800}

# Welcome to Yohou-Optuna's documentation

`OptunaSearchCV` extends Yohou's `BaseSearchCV` (which extends `BaseForecaster`) for Bayesian hyperparameter tuning of time series forecasters, powered by [Optuna](https://optuna.org/). It inherits the full yohou forecasting API (`fit(y, X, forecasting_horizon)`, `predict()`, `observe_predict()`, `best_forecaster_`, `cv_results_`) while using Optuna's samplers (TPE, CMA-ES, ...) to explore search spaces more efficiently than grid or random search. Optuna distributions give you log-scaled, bounded, and categorical parameter spaces, and wrapper classes (`Sampler`, `Storage`, `Callback`) survive `clone()` and serialization.

!!! note "Inspiration"
    This project is inspired by [optuna-integration's OptunaSearchCV](https://optuna-integration.readthedocs.io/en/latest/reference/generated/optuna_integration.OptunaSearchCV.html) and builds on [sklearn-optuna](https://github.com/stateful-y/sklearn-optuna).

<div class="grid cards" markdown>

-  **Get Started in 5 Minutes**

    ---

    Install Yohou-Optuna and run your first hyperparameter search

    Install - Define distributions - Fit - Predict

    [Getting Started](pages/tutorials/getting-started.md)

- **Understand the Design**

    ---

    Understand OptunaSearchCV, the object model, temporal CV, and wrapper classes

    [About OptunaSearchCV](pages/explanation/concepts.md)

- **See It In Action**

    ---

    Explore 5 interactive notebooks from quickstart to multi-metric search

    [Examples](pages/tutorials/examples.md)

- **API Reference**

    ---

    Complete API documentation for OptunaSearchCV and wrapper classes

    [API Reference](pages/reference/api.md)


</div>

## Documentation

### [Getting Started](pages/tutorials/getting-started.md)

A step-by-step tutorial to install Yohou-Optuna and run your first Bayesian hyperparameter search.

### [Examples](pages/tutorials/examples.md)

Interactive marimo notebooks demonstrating real-world time series hyperparameter tuning.

### [How-to Guides](pages/how-to/configure.md)

Task-focused guides for configuring `OptunaSearchCV`, running multi-metric searches, tuning composed forecasters, and more.

### [API Reference](pages/reference/api.md)

Complete API documentation for `OptunaSearchCV`, `Sampler`, `Storage`, and `Callback`.

## License

Yohou-Optuna is released under the **BSD 3-Clause License**. See the full license text in the repository.
