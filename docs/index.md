![](assets/logo_dark.png#only-dark){width=800}
![](assets/logo_light.png#only-light){width=800}

# Welcome to Yohou-Optuna's documentation

`OptunaSearchCV` extends Yohou's `BaseSearchCV` — which extends `BaseForecaster` — for Bayesian hyperparameter tuning of time series forecasters, powered by [Optuna](https://optuna.org/). It inherits the full yohou forecasting API (`fit(y, X, forecasting_horizon)`, `predict()`, `update_predict()`, `best_forecaster_`, `cv_results_`) while using Optuna's samplers (TPE, CMA-ES, …) to explore search spaces more efficiently than grid or random search. Optuna distributions give you log-scaled, bounded, and categorical parameter spaces, and wrapper classes (`Sampler`, `Storage`, `Callback`) survive `clone()` and serialization.

!!! note "Inspiration"
    This project is inspired by [optuna-integration's OptunaSearchCV](https://optuna-integration.readthedocs.io/en/latest/reference/generated/optuna_integration.OptunaSearchCV.html) and builds on [sklearn-optuna](https://github.com/stateful-y/sklearn-optuna).

<div class="grid cards" markdown>

-  **Get Started in 5 Minutes**

    ---

    Install Yohou-Optuna and run your first hyperparameter search

    Install → Define distributions → Fit → Predict

    [Getting Started](pages/getting-started.md)

- **Learn the Concepts**

    ---

    Understand OptunaSearchCV, samplers, distributions, and callbacks

    [User Guide](pages/user-guide.md)

- **See It In Action**

    ---

    Explore 5 interactive notebooks from quickstart to multi-metric search

    [Examples](pages/examples.md)

- **API Reference**

    ---

    Complete API documentation for OptunaSearchCV and wrapper classes

    [API Reference](pages/api-reference.md)


</div>

## Table of Contents

### [Getting started](pages/getting-started.md)

Step-by-step guide to installing and setting up Yohou-Optuna in your project.

- [1. Install the package](pages/getting-started.md#step-1-install-the-package)
- [2. Verify installation](pages/getting-started.md#step-2-verify-installation)
- [3. Basic usage](pages/getting-started.md#basic-usage)


### [Examples](pages/examples.md)

Interactive notebooks demonstrating real-world time series hyperparameter tuning.

- [What can Yohou-Optuna do?](pages/examples.md#what-can-yohou-optuna-do)
- [Running examples locally](pages/examples.md#running-examples-locally)


### [User guide](pages/user-guide.md)

In-depth documentation on the design, architecture, and core concepts.

- [Core Concepts](pages/user-guide.md#core-concepts)
- [Configuration](pages/user-guide.md#configuration)
- [Best Practices](pages/user-guide.md#best-practices)

### [Reference](pages/api-reference.md)

Detailed reference for the Yohou-Optuna API, including classes, functions, and configuration options.

## License

Yohou-Optuna is released under the **BSD 3-Clause License**. See the full license text in the repository.
