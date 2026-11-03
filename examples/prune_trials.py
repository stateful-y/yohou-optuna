# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "numpy",
#     "optuna",
#     "plotly",
#     "polars",
#     "scikit-learn",
#     "yohou",
#     "yohou-optuna",
# ]
# ///

import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium")

__gallery__ = {
    "title": "How to Prune Unpromising Trials",
    "description": "Stop a trial after a few backtest windows when it is reliably worse than the best one, and choose which windows run first.",
    "category": "how-to",
    "companion": "pages/how-to/prune-trials.md",
}


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # How to Prune Unpromising Trials

        Pass a `Pruner` to `OptunaSearchCV` so a trial that is reliably worse than the best one stops after a few backtest windows, instead of evaluating all of them.

        **Prerequisites**: familiarity with [`OptunaSearchCV`](/pages/api/generated/yohou_optuna.search.OptunaSearchCV/) (see [OptunaSearchCV Quickstart](/examples/optuna_search/)).
        """
    )
    return


@app.cell(hide_code=True)
def _():
    import logging
    import warnings
    from datetime import datetime, timedelta

    import numpy as np
    import optuna
    import polars as pl
    from optuna.distributions import FloatDistribution
    from optuna.exceptions import ExperimentalWarning
    from sklearn.linear_model import Ridge

    from yohou.metrics import MeanAbsoluteError
    from yohou.model_selection import ExpandingWindowSplitter
    from yohou.point import PointReductionForecaster
    from yohou.preprocessing import LagTransformer
    from yohou_optuna import OptunaSearchCV, Pruner, Sampler

    # Optuna marks WilcoxonPruner, the default pruner, as experimental; the notice is informational.
    warnings.simplefilter("ignore", ExperimentalWarning)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    logging.getLogger("yohou_optuna").setLevel(logging.ERROR)

    return (
        ExpandingWindowSplitter,
        FloatDistribution,
        LagTransformer,
        MeanAbsoluteError,
        OptunaSearchCV,
        PointReductionForecaster,
        Pruner,
        Ridge,
        Sampler,
        datetime,
        np,
        optuna,
        pl,
        timedelta,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 1. Build a Series with Enough Backtest Windows

        The default pruner needs at least four folds before it can stop a trial, so this search scores every trial over eight 12-hour windows of an hourly series with a daily cycle.
        """
    )
    return


@app.cell
def _(datetime, np, pl, timedelta):
    n_hours = 400
    start = datetime(2024, 1, 1)
    hours = np.arange(n_hours)
    rng = np.random.default_rng(0)
    y = pl.DataFrame(
        {
            "time": pl.datetime_range(start, start + timedelta(hours=n_hours - 1), "1h", eager=True),
            "load": 50 + 10 * np.sin(2 * np.pi * hours / 24) + 0.02 * hours + rng.normal(0, 1.0, n_hours),
        }
    )
    return (y,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 2. Enable Pruning

        `Pruner()` wraps Optuna's `WilcoxonPruner`. The default `fold_order="recent_first"` evaluates the most recent window first, so an early verdict rests on the windows closest to deployment.
        """
    )
    return


@app.cell
def _(
    ExpandingWindowSplitter,
    FloatDistribution,
    LagTransformer,
    MeanAbsoluteError,
    OptunaSearchCV,
    PointReductionForecaster,
    Pruner,
    Ridge,
    Sampler,
    optuna,
    y,
):
    search = OptunaSearchCV(
        forecaster=PointReductionForecaster(estimator=Ridge(), actual_transformer=LagTransformer(lag=[1, 2, 24])),
        param_distributions={"estimator__alpha": FloatDistribution(1e-3, 1e5, log=True)},
        scoring=MeanAbsoluteError(),
        sampler=Sampler(sampler=optuna.samplers.TPESampler, seed=0),
        n_trials=20,
        cv=ExpandingWindowSplitter(n_splits=8, test_size=12),
        pruner=Pruner(),
        refit=False,
    )
    search.fit(y, forecasting_horizon=12)
    assert search.n_pruned_ > 0, "the demo search should prune at least one trial"
    return (search,)


@app.cell(hide_code=True)
def _(mo, search):
    mo.md(
        f"""
        Trials run: {len(search.trials_)}. Pruned: {search.n_pruned_}. Rows in `cv_results_`: {len(search.cv_results_["params"])}.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 3. Read the Pruned Trials

        Pruned trials have no row in `cv_results_`. Read them from `trials_`: each keeps the fold scores it reported, keyed by the fold's index in the splitter, and no mean.
        """
    )
    return


@app.cell
def _(optuna, pl, search):
    pruned = [t for t in search.trials_ if t.state == optuna.trial.TrialState.PRUNED]
    pl.DataFrame(
        {
            "trial": [t.number for t in pruned],
            "alpha": [t.params["estimator__alpha"] for t in pruned],
            "folds_run": [sorted(t.intermediate_values) for t in pruned],
            "has_mean": ["mean_test_score" in t.user_attrs for t in pruned],
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 4. Plot Fold Scores per Trial

        Each line is one trial across its folds, numbered oldest first. A pruned trial ran its most recent windows first, so its line covers only the highest fold numbers and is missing the oldest ones.
        """
    )
    return


@app.cell
def _(optuna, search):
    optuna.visualization.plot_intermediate_values(search.study_)
    return


if __name__ == "__main__":
    app.run()
