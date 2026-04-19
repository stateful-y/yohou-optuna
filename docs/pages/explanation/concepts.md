# About OptunaSearchCV

`OptunaSearchCV` is Yohou's integration of Optuna's Bayesian hyperparameter optimization. This page explores how it fits into Yohou's object model, how the search operates internally, and how the design connects to the broader ecosystem.

## The Object Model

`OptunaSearchCV` inherits from Yohou's `BaseSearchCV`, which itself inherits from `BaseForecaster`. This means a fitted `OptunaSearchCV` **is** a forecaster, exposing the complete Yohou forecaster interface: `fit()`, `predict()`, `predict_interval()`, `observe()`, and `observe_predict()`.

After fitting, you never need to unwrap it to access the best forecaster; `predict()` delegates to `best_forecaster_` automatically. This follows the sklearn convention that fitted search objects behave as the estimators they select. In Yohou, where the forecaster API carries temporal semantics (`fit(y, X, forecasting_horizon)`, observation windows, panel data), this inheritance is particularly valuable: `OptunaSearchCV` participates in the same compositions, pipelines, and evaluation loops as any other forecaster.

`OptunaSearchCV` works with both point forecasters (e.g., `PointReductionForecaster`) and interval forecasters (e.g., `SplitConformalForecaster`). When interval scorers are used, coverage rates are routed automatically and prediction calls `predict_interval()` instead of `predict()`.

## The Search Lifecycle

Calling `fit(y, X, forecasting_horizon)` triggers the following steps:

1. An Optuna `Study` is created (or loaded from storage if `study_name` and `storage` are provided).
2. For each of `n_trials` trials, the sampler proposes a parameter combination from `param_distributions`.
3. A clone of the base forecaster is created with those parameters and evaluated via cross-validation. The mean CV score across folds becomes the trial's objective value.
4. Optuna records the result and the sampler updates its internal model of the search space.
5. After all trials, the best parameter combination is used to refit the forecaster on the full training data. The fitted forecaster is stored as `best_forecaster_`.

The study object (`study_`) remains accessible after fitting. You can inspect the full trial history, visualize parameter importance, or resume the study in a later `fit()` call by passing `study=search.study_`.

## Temporal Cross-Validation

Hyperparameter search for time series differs from iid settings: using future data to evaluate parameters selected for past data inflates estimates and leads to poor generalization.

`OptunaSearchCV` delegates cross-validation to Yohou's splitter API. By default it uses a 5-fold expanding window split. You can supply any splitter from `yohou.model_selection`, such as `ExpandingWindowSplitter`, `SlidingWindowSplitter`, or a custom subclass. The splitter determines how the training data is partitioned into fold (past) and validation (future) windows, preserving temporal ordering throughout the search.

This is why the `cv` parameter accepts Yohou splitters rather than sklearn cross-validators: time series folds are defined by their position in time, not by random index shuffles.

## Wrapper Classes and Cloneability

Optuna's sampler, storage, and callback objects are not compatible with Python's `copy.deepcopy()`, which sklearn's `clone()` uses internally. `BaseSearchCV` calls `clone()` to create fresh copies of the forecaster for each trial. Passing an `optuna.samplers.TPESampler` directly would fail at the first cloning step.

The `Sampler`, `Storage`, and `Callback` wrappers (re-exported from [sklearn-optuna](https://github.com/stateful-y/sklearn-optuna)) solve this by storing the class name and constructor arguments rather than the live Optuna object. When the study needs to be created, the wrapper instantiates the underlying Optuna object on demand. Since a wrapper holds only serializable Python values, `clone()` can copy it safely.

```python
# Wrappers hold arguments, not live Optuna objects
sampler = Sampler("TPESampler", seed=42)
storage = Storage("RDBStorage", url="sqlite:///study.db")
callback = Callback("MaxTrialsCallback", n_trials=100)
```

The lazy instantiation also means the Optuna backend (database connections, sampler state) is only initialized when the study is actually created, not when you construct `OptunaSearchCV`.

## Parameter Distributions

Optuna's distribution objects define the search space for each parameter. Unlike a fixed grid (exhaustive) or a uniform random range (uninformed), distributions carry structure that the sampler exploits. A log-scaled float distribution tells the sampler that a regularization strength spanning 0.0001 to 10 should be explored on a logarithmic scale, focusing candidates evenly across orders of magnitude. An integer distribution signals that only whole numbers are valid, and a categorical distribution signals unordered choices. This type information lets the sampler allocate its budget more efficiently than it could with a flat grid or unstructured range.

Nested parameters in composed forecasters use the double-underscore routing convention from sklearn: `"deseason__regressor__alpha"` routes `alpha` to the regressor inside the `deseason` step. See the [API Reference](../reference/api.md) for the full list of distribution types.

## Samplers and Adaptive Search

Optuna's default sampler, TPE (Tree-structured Parzen Estimator), builds a probabilistic model of the objective function from observed trials and uses it to propose the next candidate. Unlike random search, which treats all candidates uniformly, TPE focuses exploration on regions that performed well in past trials. This typically reaches good solutions in fewer evaluations.

Other samplers are available: CMA-ES for continuous search spaces with correlated parameters, Gaussian Process sampler for small budgets with smooth objectives, and Random sampler as a baseline. The sampler is swappable via the `sampler` parameter without changing any other part of the code.

## See Also

- [Getting Started](../tutorials/getting-started.md) - hands-on walkthrough of your first search
- [Configure OptunaSearchCV](../how-to/configure.md) - samplers, callbacks, and storage options
- [API Reference](../reference/api.md) - full parameter and attribute documentation
