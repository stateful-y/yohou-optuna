# How to Configure OptunaSearchCV

This guide shows you how to configure `OptunaSearchCV` for common search scenarios: changing the sampler, adding callbacks, persisting studies, and customizing cross-validation.

## Prerequisites

- Yohou-Optuna installed ([Getting Started](../tutorials/getting-started.md))
- Familiarity with `OptunaSearchCV.fit()` basics

## Choose a Sampler

The `sampler` parameter controls the optimization strategy. The default (TPE) works well for most cases. Wrap any Optuna sampler with the `Sampler` class to make it compatible with `get_params()` / `set_params()` / `clone()`.

```python
import optuna
from yohou_optuna import OptunaSearchCV, Sampler

search = OptunaSearchCV(
    forecaster=forecaster,
    param_distributions=distributions,
    scoring=scorer,
    n_trials=50,
    sampler=Sampler(sampler=optuna.samplers.TPESampler, seed=42),
)
```

For other strategies, swap the sampler class:

```python
# CMA-ES: effective for continuous spaces with correlated parameters
sampler=Sampler(sampler=optuna.samplers.CmaEsSampler, seed=42)

# Gaussian Process: best for very small budgets (< 20 trials)
sampler=Sampler(sampler=optuna.samplers.GPSampler)

# Random: useful as a baseline or for reproducible ablations
sampler=Sampler(sampler=optuna.samplers.RandomSampler, seed=42)
```

Pass `seed` for reproducible results when `n_jobs=1`. Always use the `Sampler` wrapper rather than a raw Optuna sampler object because raw Optuna objects are not compatible with `clone()`.

!!! tip
    Start with TPE. Switch to CMA-ES only when you have a large all-continuous search space and notice slow convergence.

## Add Callbacks

Callbacks run after each completed trial. Use them for early stopping, logging, or custom logic. Pass a dictionary mapping callback names to `Callback` instances:

```python
from optuna.study import MaxTrialsCallback
from yohou_optuna import Callback

search = OptunaSearchCV(
    forecaster=forecaster,
    param_distributions=distributions,
    scoring=scorer,
    n_trials=200,
    callbacks={
        "stop": Callback(callback=MaxTrialsCallback, n_trials=50),
    },
)
```

`MaxTrialsCallback` stops the study once the specified number of trials completes, regardless of the `n_trials` setting on `OptunaSearchCV`. This is useful when you want to set a generous upper bound on trials but stop early once you have enough results.

Always use the `Callback` wrapper instead of a raw Optuna callback for the same cloneability reasons as `Sampler`.

## Write a Custom Callback

Any callable class that accepts `study` and `trial` arguments works as a callback:

```python
class EarlyStoppingCallback:
    def __init__(self, patience: int = 10):
        self.patience = patience

    def __call__(self, study, trial):
        if trial.number >= self.patience:
            best = study.best_trial.number
            if trial.number - best >= self.patience:
                study.stop()

search = OptunaSearchCV(
    forecaster=forecaster,
    param_distributions=distributions,
    scoring=scorer,
    n_trials=200,
    callbacks={
        "early_stop": Callback(callback=EarlyStoppingCallback, patience=10),
    },
)
```

## Persist and Resume Studies

For long-running or distributed searches, save the study to a storage backend:

```python
import optuna
from yohou_optuna import Storage

search = OptunaSearchCV(
    forecaster=forecaster,
    param_distributions=distributions,
    scoring=scorer,
    n_trials=50,
    storage=Storage(storage=optuna.storages.RDBStorage, url="sqlite:///my_study.db"),
)

search.fit(y_train, forecasting_horizon=12)
```

To add more trials later, pass the existing study back to `fit()`:

```python
search.n_trials = 50  # 50 additional trials
search.fit(y_train, forecasting_horizon=12, study=search.study_)
```

To name a study for easier identification, create it externally and pass it via `fit()`:

```python
study = optuna.create_study(
    study_name="ridge_air_passengers",
    direction="maximize",
    storage="sqlite:///my_study.db",
)
search.fit(y_train, forecasting_horizon=12, study=study)
```

## Use a Custom CV Splitter

By default, `OptunaSearchCV` uses a 5-fold expanding window. Pass any Yohou splitter to change this:

```python
from yohou.model_selection import SlidingWindowSplitter

search = OptunaSearchCV(
    forecaster=forecaster,
    param_distributions=distributions,    scoring=scorer,    n_trials=30,
    cv=SlidingWindowSplitter(n_splits=5, train_size=24),
)
```

Use `ExpandingWindowSplitter` (default) for growing training windows. Use `SlidingWindowSplitter` for a fixed-size training window.

## Handle Fitting Errors

By default, `error_score=np.nan` absorbs errors during cross-validation folds. A failed fold contributes `error_score` to a plain mean over every fold, so with the `NaN` default a trial with any failed fold scores `NaN`; a trial is never scored on only the folds it survived. Each absorbed failure is also recorded on the trial (the `exception`, `exception_type`, and `failed_splits` user attributes) and logged as a warning naming the trial and the failed splits. To stop the search immediately on the first error instead, set `error_score="raise"`:

```python
search = OptunaSearchCV(
    forecaster=forecaster,
    param_distributions=distributions,
    scoring=scorer,
    n_trials=50,
    error_score="raise",  # stop on first error (useful during development)
)
```

Use `error_score="raise"` during development to catch bad parameter combinations early. In production, keep the default (`np.nan`) so the search continues past occasional failures.

## Filter Failed Trials

After fitting, inspect which trials failed:

```python
import polars as pl

results = pl.DataFrame(search.cv_results_)
failed = results.filter(pl.col("mean_test_score").is_nan())
print(f"{len(failed)} trials failed out of {len(results)}")
```

The search also exposes the counts directly: `search.n_scored_` is the number of completed trials with a finite objective, alongside `search.n_completed_`. A trial that carries absorbed fold failures marks them in its `failed_splits` user attribute, so a failure is distinguishable from a genuinely poor score even when `error_score` is a finite number:

```python
for trial in search.trials_:
    if "failed_splits" in trial.user_attrs:
        print(trial.number, trial.user_attrs["exception_type"], trial.user_attrs["failed_splits"])
```

## Collect Training Scores

To compute scores on the training folds in addition to the validation folds, set `return_train_score=True`:

```python
search = OptunaSearchCV(
    forecaster=forecaster,
    param_distributions=distributions,
    scoring=scorer,
    n_trials=30,
    return_train_score=True,
)

search.fit(y_train, forecasting_horizon=12)

# cv_results_ now contains train_score columns alongside test_score columns
import polars as pl
results = pl.DataFrame(search.cv_results_)
print(results.select(["params", "mean_test_score", "mean_train_score"]))
```

Large gaps between training and test scores suggest overfitting.

Training scores are computed by yohou's own train-score recipe, so a trial's `split{i}_train_score` equals what yohou's `cross_validate(..., return_train_score=True)` reports for the same forecaster, parameters and split:

- **Which rows are scored.** Each split's training score covers a stretch as long as the test window, predicted the way the test window is (the fitted forecaster is rewound and walked forward, without refitting). The stretch ends before any rows the forecaster holds back from learning, which it declares in its `holdout_size` tag. For a `SplitConformalForecaster` those are its `calibration_size` calibration rows, so its training score measures the point forecaster on data it was fitted on, not the rows that sized its intervals. For a reduction forecaster with a `validation_size`, they are the early-stopping holdout at the end of each fold's training window.
- **Short training windows.** When a split's training window is no longer than the test window plus the held-back rows, that split's training score is `NaN` and yohou emits a `UserWarning` naming the three lengths. This is common on the first splits of an expanding window with a large `test_size`. `mean_train_score` then rests on fewer splits than `mean_test_score`.
- **Failures.** An error while computing a training score is handled like any other fold failure: it raises under `error_score="raise"`, and otherwise the split is recorded in the trial's `failed_splits`.

See yohou's [model selection explanation](https://yohou.readthedocs.io/en/latest/pages/explanation/model-selection/) for why the training score is measured this way.

## Early-Stop Boosting Estimators

A reduction forecaster wrapping a gradient boosting estimator (LightGBM, XGBoost, CatBoost, or scikit-learn's histogram gradient boosting) stops early only when its `fit` receives an evaluation set. There are two ways to give it one inside a search.

**Hold out a tail in every fold** with the forecaster's own `validation_size`. Each fold's fit sets its last `validation_size` rows aside as the evaluation set and stops on its own patience. It needs no search setting, and the fold's test window is never used to choose the round, so the scores stay unbiased:

```python
from lightgbm import LGBMRegressor
from yohou.point import PointReductionForecaster
from yohou.preprocessing import LagTransformer

forecaster = PointReductionForecaster(
    estimator=LGBMRegressor(n_estimators=1000, early_stopping_round=20, verbose=-1),
    reduction_strategy="direct",
    actual_transformer=LagTransformer(lag=[1, 2, 24]),
    validation_size=96,
)
```

**Share one round across folds** with `validation="cv"`, the mode yohou's `GridSearchCV` and `RandomizedSearchCV` provide. Every fold of a trial uses its own test window as the evaluation set and trains every round up to the estimator's ceiling. One round per fitted estimator is chosen from the stopping metric averaged over the folds, and every fold is scored cut to that round. The refit then trains exactly that many rounds on all the data, with early stopping off:

```python
from lightgbm import LGBMRegressor
from optuna.distributions import FloatDistribution, IntDistribution
from yohou.point import PointReductionForecaster
from yohou.preprocessing import LagTransformer
from yohou_optuna import OptunaSearchCV

search = OptunaSearchCV(
    forecaster=PointReductionForecaster(
        estimator=LGBMRegressor(n_estimators=1000, verbose=-1),
        reduction_strategy="direct",
        actual_transformer=LagTransformer(lag=[1, 2, 24]),
    ),
    param_distributions={
        "estimator__num_leaves": IntDistribution(8, 128),
        "estimator__learning_rate": FloatDistribution(0.01, 0.2, log=True),
    },
    scoring=scorer,
    n_trials=30,
    validation="cv",
)
search.fit(y_train, forecasting_horizon=12)

search.best_rounds_                  # chosen round per fitted estimator, e.g. {"step_1": 412, ...}
search.cv_results_["rounds"]         # each trial's chosen rounds
search.cv_results_["rounds_at_boundary"]  # True when a round hit the ceiling
```

A few things to know about `validation="cv"`:

- **The scores are optimistic.** The round is chosen on the same rows that produce the trial's score, as if `n_estimators` were searched on the test folds. Prefer `validation_size` when the score itself matters, and use `validation="cv"` to find a round count for the refit.
- **The ceiling is the cost.** Folds train to `n_estimators` (or `iterations`, or `max_iter`), which is also the largest round the search can choose. A trial whose chosen round sits at the ceiling sets `rounds_at_boundary` and emits a warning: raise the ceiling.
- **Some configurations are refused, and stop the search.** yohou rejects non-reduction forecasters, a forecaster that sets `validation_size`, `reduction_strategy="dir-rec"`, LightGBM or XGBoost dart boosting, and CatBoost without an explicit `learning_rate`. A forecaster or fit parameter it rejects fails `fit` before any trial runs. A sampled parameter it rejects, such as `reduction_strategy` drawn as `"dir-rec"`, raises and stops the study rather than being recorded as a failed trial, the way the same configuration stops `GridSearchCV`. Keep such values out of `param_distributions`.
- **Fold failures still follow `error_score`**, exactly as without `validation="cv"`: a fold whose fit raises contributes `error_score`, is listed in the trial's `failed_splits`, and adds no stopping curve.

Pass `early_stopping_adapter=` to use a custom `BaseEarlyStoppingAdapter` for an estimator yohou has no built-in adapter for. See yohou's [early stopping how-to](https://yohou.readthedocs.io/en/latest/pages/how-to/early-stopping/) for how the estimator is configured and how adapters work.

## See Also

- [About OptunaSearchCV](../explanation/concepts.md): understand samplers, temporal CV, and wrapper classes
- yohou's [early stopping how-to](https://yohou.readthedocs.io/en/latest/pages/how-to/early-stopping/): configuring the estimator, `validation_size`, and `validation="cv"` in yohou's own searches
- [Multi-Metric Search](multi-metric-search.md): evaluate multiple metrics simultaneously
- [API Reference](../reference/api.md): full parameter documentation for `OptunaSearchCV`, `Sampler`, `Storage`, `Callback`
