# How to Prune Unpromising Trials

This guide shows you how to stop a trial after a few backtest folds when it is reliably worse than the best trial so far, instead of evaluating every fold. Use this when fits are expensive and your search scores each trial over several backtest windows.

## Prerequisites

- Yohou-Optuna installed ([Getting Started](../tutorials/getting-started.md))
- A working `OptunaSearchCV` search
- At least four backtest folds (see step 2)

<!-- COMPANION_NOTEBOOKS -->

## 1. Enable Pruning

Pass a `Pruner` wrapper. With no arguments it wraps Optuna's `WilcoxonPruner`, which compares the trial's fold scores with the best trial's scores on the same folds:

```python
from yohou.model_selection import ExpandingWindowSplitter
from yohou_optuna import OptunaSearchCV, Pruner

search = OptunaSearchCV(
    forecaster=forecaster,
    param_distributions=distributions,
    scoring=scorer,
    n_trials=50,
    cv=ExpandingWindowSplitter(n_splits=8, test_size=24),
    pruner=Pruner(),
)
search.fit(y_train, forecasting_horizon=24)
```

Optuna marks `WilcoxonPruner` as experimental and emits an `ExperimentalWarning` when it is created. The warning is informational.

## 2. Give the Pruner Enough Folds

The default pruner cannot stop a trial before its fourth fold. With three folds it never prunes, and `fit()` warns:

```text
UserWarning: The pruner cannot stop any trial: WilcoxonPruner with p_threshold=0.1
and n_startup_steps=2 needs at least 4 folds, but cv yields 3.
```

If you see this warning, raise `n_splits` on the splitter. The default `cv` gives five folds, which is enough.

## 3. Choose Which Folds Run First

`fold_order` decides which backtest windows a trial evaluates first, and so which ones a pruner judges first:

- Leave the default, `"recent_first"`, to judge a trial on the most recent windows.
- Pass `fold_order="chronological"` to run the oldest window first.
- Pass `fold_order="random"` with an integer `random_state` to draw a reproducible order per trial:

```python
search = OptunaSearchCV(
    forecaster=forecaster,
    param_distributions=distributions,
    scoring=scorer,
    pruner=Pruner(),
    fold_order="random",
    random_state=0,
)
```

The order never changes the scores of a trial that evaluates every fold. See [Why Fold Order Matters](../explanation/concepts.md#why-fold-order-matters).

## 4. Use a Different Pruner

Any Optuna pruner class can be wrapped. If you use one other than `WilcoxonPruner`, also pass `fold_order="chronological"`:

```python
import optuna

search = OptunaSearchCV(
    forecaster=forecaster,
    param_distributions=distributions,
    scoring=scorer,
    pruner=Pruner(pruner=optuna.pruners.MedianPruner, n_startup_trials=5, n_warmup_steps=1),
    fold_order="chronological",
)
```

With any other order, `fit()` warns that the pruner judges a trial on its highest-numbered fold so far; under `"recent_first"` that is the first fold run, and the warm-up is skipped. A warm-up at or above the number of folds also disables the median and percentile pruners, and `fit()` warns in that case too. Refer to the [Optuna pruners reference](https://optuna.readthedocs.io/en/stable/reference/pruners.html) for the available classes and their parameters.

## 5. Read the Pruned Trials

Pruned trials have no row in `cv_results_`. Count them and read them from the study:

```python
import optuna

print(search.n_pruned_, "pruned of", len(search.trials_))

pruned = [t for t in search.trials_ if t.state == optuna.trial.TrialState.PRUNED]
for trial in pruned:
    print(trial.params, trial.intermediate_values)
```

`trial.intermediate_values` maps fold index to that fold's test score. A pruned trial also keeps `split<i>_test_score`, `split<i>_fit_time`, `split<i>_score_time` and, if a fold failed, `failed_splits` for the folds that ran.

Plot the fold scores of every trial. A pruned trial's line covers only the folds it ran; under the default order those are the most recent windows, so it is missing the oldest folds:

```python
fig = optuna.visualization.plot_intermediate_values(search.study_)
fig.show()
```

## Common Variations

- **Failed folds**: under the default `error_score=np.nan`, a failed fold is not reported, so its index is missing from `intermediate_values`. With a numeric `error_score`, the error value is reported and the pruner judges it as a real score.
- **Shared-round early stopping**: `pruner` cannot be combined with `validation="cv"`, and `fit()` raises `ValueError`. Use the forecaster's `validation_size` for early stopping inside each fold instead; see [Early-Stop Boosting Estimators](configure.md#early-stop-boosting-estimators).
- **Resumed studies**: when you pass `study=`, a given `pruner` replaces the study's own pruner. A study created before pruning was enabled has no intermediate values on its earlier trials, so the default pruner stays silent until a trial run with a pruner completes.

## Troubleshooting

**Problem: `n_pruned_` is 0**
: Check for the "cannot stop any trial" warning and add folds. Pruning also needs a completed trial to compare against, and a search whose candidates all score alike has nothing to prune.

**Problem: `TypeError: pruner must be a Pruner instance`**
: You passed a raw Optuna pruner. Wrap the class instead: `Pruner(pruner=optuna.pruners.MedianPruner, n_warmup_steps=1)`.

**Problem: a parameter combination is missing from `cv_results_`**
: It was pruned. Look it up in `search.trials_`.

## See Also

- [Pruning](../explanation/concepts.md#pruning): how fold scores are paired across trials and what a pruned trial keeps
- [API Reference](../reference/api.md): the `pruner`, `fold_order` and `random_state` parameters and the `n_pruned_` attribute
- [Configure OptunaSearchCV](configure.md): samplers, callbacks, storage and custom splitters
