import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Panel Data Tuning

        ## What You'll Learn

        - How to use `OptunaSearchCV` with panel (grouped) time series data
        - How to visualize cross-validation splits on panel data with `plot_splits`
        - How to compare different Optuna samplers (Random vs TPE)
        - How to use callbacks for early stopping with `MaxTrialsCallback`
        - How to visualize panel-aware forecasts

        ## Prerequisites

        Familiarity with the basics of `OptunaSearchCV` (see optuna_search.py) and
        panel data concepts (multiple related time series sharing the same time index).
        """
    )
    return


@app.cell(hide_code=True)
async def _():
    import sys

    if "pyodide" in sys.modules:
        import micropip

        await micropip.install(["scikit-learn", "optuna", "sklearn-optuna"])
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
        ## 1. Load and Explore Panel Data

        The Australian Tourism dataset contains quarterly trip counts for 8 Australian
        states/territories. Each column follows the `group__member` naming convention
        (e.g., `victoria__trips`), which yohou recognizes as panel data.
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
        ## 2. Configure Splitter and Forecaster

        For panel data, `ExpandingWindowSplitter` applies the same temporal split
        across all groups. Use `plot_splits` to visualize the cross-validation
        strategy before running the search.
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

        `RandomSampler` explores the space uniformly, while `TPESampler` uses
        Bayesian optimization to focus on promising regions. Compare their
        performance on the same panel forecasting task.
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

        `MaxTrialsCallback` stops the search after a fixed number of trials,
        even if `n_trials` is set higher. This is useful when you want a time
        budget or want to interactively control search duration.
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
        ## 5. Inspect Panel Forecasts

        Generate forecasts with the best model from the TPE search and
        visualize them against the held-out test set. For panel data,
        `plot_forecast` shows all groups together.
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
        ## Key Takeaways

        - **Panel data** with `group__member` column naming is automatically recognized by yohou forecasters and splitters
        - **`ExpandingWindowSplitter`** applies the same temporal split across all panel groups
        - **`plot_splits`** visualizes how training and test windows expand across CV folds
        - **`RandomSampler` vs `TPESampler`**: TPE typically finds better configurations with fewer trials
        - **`MaxTrialsCallback`** provides early stopping control via the `callbacks` parameter
        - **`plot_forecast`** handles panel data by showing all groups together

        ## Next Steps

        - **Quickstart**: See optuna_search.py for a minimal end-to-end walkthrough
        - **Composed tuning**: See composed_tuning.py to tune forecasters with feature transformers
        - **Search visualization**: See search_visualization.py for Optuna's optimization plots
        """
    )
    return


if __name__ == "__main__":
    app.run()
