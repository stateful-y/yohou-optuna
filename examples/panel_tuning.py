# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "optuna",
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
    "title": "How to Tune on Panel Data",
    "description": "Apply OptunaSearchCV to grouped time series from the Australian Tourism dataset with sampler comparison.",
    "category": "how-to",
    "companion": "pages/how-to/panel-data.md",
}


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # How to Tune on Panel Data

        Run `OptunaSearchCV` on grouped time series, compare samplers, and use `MaxTrialsCallback` for early stopping.

        **Prerequisites**: familiarity with [`OptunaSearchCV`](/pages/api/generated/yohou_optuna.search.OptunaSearchCV/) (see [OptunaSearchCV Quickstart](/examples/optuna_search/)) and panel data concepts.
        """
    )
    return


@app.cell(hide_code=True)
def _():
    import optuna
    import polars as pl
    from optuna.distributions import CategoricalDistribution, FloatDistribution
    from sklearn.linear_model import Ridge

    from yohou.datasets import load_australian_tourism
    from yohou.metrics import MeanAbsoluteError
    from yohou.model_selection import ExpandingWindowSplitter
    from yohou.plotting import plot_forecast, plot_splits, plot_time_series
    from yohou.point import PointReductionForecaster

    from yohou_optuna import Callback, OptunaSearchCV, Sampler

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    return (
        Callback,
        CategoricalDistribution,
        FloatDistribution,
        MeanAbsoluteError,
        OptunaSearchCV,
        PointReductionForecaster,
        Ridge,
        Sampler,
        load_australian_tourism,
        optuna,
        pl,
        plot_forecast,
        plot_splits,
        plot_time_series,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 1. Load the Panel Data

        Load the Australian Tourism dataset (quarterly trips for 8 states, using `group__member` columns).
        """
    )
    return


@app.cell
def _(load_australian_tourism, plot_time_series):
    y = load_australian_tourism()
    plot_time_series(y, title="Australian Tourism: Quarterly Trips by State")
    return (y,)


@app.cell
def _(y):
    y_train = y.head(60)
    y_test = y.tail(20)
    return y_test, y_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 2. Configure the Splitter

        Set up an `ExpandingWindowSplitter` and visualize the CV strategy with `plot_splits`.
        """
    )
    return


@app.cell
def _(ExpandingWindowSplitter):
    cv = ExpandingWindowSplitter(n_splits=3, test_size=4)
    return (cv,)


@app.cell
def _(cv, plot_splits, y_train):
    plot_splits(y_train, cv, title="Expanding Window CV Splits")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 3. Compare Samplers: Random vs TPE

        Run two searches with different samplers on the same task to compare convergence.
        """
    )
    return


@app.cell
def _(
    CategoricalDistribution,
    FloatDistribution,
    MeanAbsoluteError,
    OptunaSearchCV,
    PointReductionForecaster,
    Ridge,
    Sampler,
    cv,
    optuna,
    y_train,
):
    param_distributions = {
        "estimator__alpha": FloatDistribution(0.001, 100.0, log=True),
        "estimator__fit_intercept": CategoricalDistribution([True, False]),
    }

    random_search = OptunaSearchCV(
        forecaster=PointReductionForecaster(estimator=Ridge()),
        param_distributions=param_distributions,
        scoring=MeanAbsoluteError(),
        sampler=Sampler(sampler=optuna.samplers.RandomSampler, seed=42),
        n_trials=15,
        cv=cv,
        refit=True,
        verbose=0,
    )
    random_search.fit(y_train, forecasting_horizon=4)
    return param_distributions, random_search


@app.cell
def _(
    MeanAbsoluteError,
    OptunaSearchCV,
    PointReductionForecaster,
    Ridge,
    Sampler,
    cv,
    optuna,
    param_distributions,
    y_train,
):
    tpe_search = OptunaSearchCV(
        forecaster=PointReductionForecaster(estimator=Ridge()),
        param_distributions=param_distributions,
        scoring=MeanAbsoluteError(),
        sampler=Sampler(sampler=optuna.samplers.TPESampler, seed=42),
        n_trials=15,
        cv=cv,
        refit=True,
        verbose=0,
    )
    tpe_search.fit(y_train, forecasting_horizon=4)
    return (tpe_search,)


@app.cell
def _(mo, random_search, tpe_search):
    mo.md(
        f"""
        | Sampler | Best MAE | Best Parameters |
        |---------|----------|-----------------|
        | Random  | `{random_search.best_score_:.4f}` | `{random_search.best_params_}` |
        | TPE     | `{tpe_search.best_score_:.4f}` | `{tpe_search.best_params_}` |
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 4. Use Callbacks for Early Stopping

        Pass a `MaxTrialsCallback` to stop the search after a fixed number of trials,
        regardless of `n_trials`.
        """
    )
    return


@app.cell
def _(
    Callback,
    MeanAbsoluteError,
    OptunaSearchCV,
    PointReductionForecaster,
    Ridge,
    Sampler,
    cv,
    optuna,
    param_distributions,
    y_train,
):
    callback_search = OptunaSearchCV(
        forecaster=PointReductionForecaster(estimator=Ridge()),
        param_distributions=param_distributions,
        scoring=MeanAbsoluteError(),
        sampler=Sampler(sampler=optuna.samplers.TPESampler, seed=42),
        n_trials=100,
        cv=cv,
        refit=True,
        verbose=0,
        callbacks={
            "max_trials": Callback(
                callback=optuna.study.MaxTrialsCallback, n_trials=8
            )
        },
    )
    callback_search.fit(y_train, forecasting_horizon=4)
    return (callback_search,)


@app.cell
def _(callback_search, mo):
    mo.md(
        f"""
        **Completed trials:** `{len(callback_search.trials_)}` (stopped early by MaxTrialsCallback)

        **Best score:** `{callback_search.best_score_:.4f}`
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 5. Generate a Panel Forecast

        Predict with the best TPE model and compare against the held-out test set.
        """
    )
    return


@app.cell
def _(tpe_search):
    y_pred = tpe_search.predict(forecasting_horizon=4)
    y_pred
    return (y_pred,)


@app.cell
def _(plot_forecast, y_pred, y_test, y_train):
    plot_forecast(
        y_test,
        y_pred,
        y_train=y_train,
        n_history=12,
        title="Panel Forecast: Australian Tourism",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Next Steps

        - [OptunaSearchCV Quickstart](/examples/optuna_search/): minimal end-to-end walkthrough
        - [How to Tune Composed Forecasters](/examples/composed_tuning/): tune forecasters with feature transformers
        - [How to Visualize Search Results](/examples/search_visualization/): Optuna's optimization plots
        """
    )
    return


if __name__ == "__main__":
    app.run()
