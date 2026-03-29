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
    "title": "How to Visualize Search Results",
    "description": "Combine Optuna's optimization plots with yohou's forecast diagnostics for comprehensive analysis.",
    "category": "how-to",
    "companion": "pages/how-to/visualize-study.md",
}


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # How to Visualize Search Results

        Access the Optuna `study_` object from a fitted `OptunaSearchCV` and produce optimization history, parameter importance, contour, and slice plots alongside yohou's forecast diagnostics.

        **Prerequisites** - familiarity with [`OptunaSearchCV`](/pages/api/generated/yohou_optuna.search.OptunaSearchCV/) (see [OptunaSearchCV Quickstart](/examples/optuna_search/)).
        """
    )
    return


@app.cell(hide_code=True)
def _():
    import optuna
    import polars as pl
    from optuna.distributions import CategoricalDistribution, FloatDistribution
    from sklearn.linear_model import Ridge

    from yohou.datasets import load_vic_electricity
    from yohou.metrics import MeanAbsoluteError
    from yohou.plotting import (
        plot_cv_results_scatter,
        plot_forecast,
        plot_residual_time_series,
        plot_time_series,
    )
    from yohou.point import PointReductionForecaster

    from yohou_optuna import OptunaSearchCV, Sampler

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    return (
        CategoricalDistribution,
        FloatDistribution,
        MeanAbsoluteError,
        OptunaSearchCV,
        PointReductionForecaster,
        Ridge,
        Sampler,
        load_vic_electricity,
        optuna,
        pl,
        plot_cv_results_scatter,
        plot_forecast,
        plot_residual_time_series,
        plot_time_series,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 1. Load the Data

        Load a subset of the Victoria Electricity demand series.
        """
    )
    return


@app.cell
def _(load_vic_electricity, pl):
    y_full = load_vic_electricity()
    y_all = y_full.select(["time", "Demand"]).head(500)
    plot_time_series_fig = None
    return y_all, y_full


@app.cell
def _(plot_time_series, y_all):
    plot_time_series(y_all, title="Victoria Electricity Demand (Subset)")
    return


@app.cell
def _(y_all):
    y_train = y_all.head(400)
    y_test = y_all.tail(24)
    return y_test, y_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 2. Run the Search

        Search over two parameters with 25 trials to generate data for the plots.
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
    optuna,
    y_train,
):
    search = OptunaSearchCV(
        forecaster=PointReductionForecaster(estimator=Ridge()),
        param_distributions={
            "estimator__alpha": FloatDistribution(0.001, 100.0, log=True),
            "estimator__fit_intercept": CategoricalDistribution([True, False]),
        },
        scoring=MeanAbsoluteError(),
        sampler=Sampler(sampler=optuna.samplers.TPESampler, seed=42),
        n_trials=25,
        cv=3,
        refit=True,
        verbose=0,
    )

    search.fit(y_train, forecasting_horizon=24)
    return (search,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 3. Optuna Study Visualizations

        Access the underlying Optuna study via `search.study_`.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Optimization History** - objective value across trials with a running best line.
        """
    )
    return


@app.cell
def _(optuna, search):
    optuna.visualization.plot_optimization_history(search.study_)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Parameter Importances** - contribution of each hyperparameter to objective variation.
        """
    )
    return


@app.cell
def _(optuna, search):
    optuna.visualization.plot_param_importances(search.study_)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Slice Plots** - per-parameter objective values across trials.
        """
    )
    return


@app.cell
def _(optuna, search):
    optuna.visualization.plot_slice(search.study_)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Contour Plot** - joint parameter interaction as a 2D surface.
        """
    )
    return


@app.cell
def _(optuna, search):
    optuna.visualization.plot_contour(
        search.study_,
        params=["estimator__alpha", "estimator__fit_intercept"],
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 4. Yohou Forecast Diagnostics

        Use yohou's plotting module for forecast-level analysis.
        """
    )
    return


@app.cell
def _(plot_cv_results_scatter, search):
    plot_cv_results_scatter(
        search.cv_results_,
        param_name="estimator__alpha",
        higher_is_better=False,
        title="CV Score vs Regularization Strength",
    )
    return


@app.cell
def _(search):
    y_pred = search.predict(forecasting_horizon=24)
    y_pred
    return (y_pred,)


@app.cell
def _(plot_forecast, y_pred, y_test, y_train):
    plot_forecast(
        y_test,
        y_pred,
        y_train=y_train,
        n_history=48,
        title="Best Forecaster: Predicted vs Actual",
    )
    return


@app.cell
def _(plot_residual_time_series, y_pred, y_test):
    plot_residual_time_series(
        y_pred,
        y_test,
        title="Forecast Residuals",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Next Steps

        - [How to Run a Multi-Metric Search](/examples/multi_metric_search/) - track multiple metrics and compare rankings
        - [How to Tune on Panel Data](/examples/panel_tuning/) - grouped time series optimization
        - [OptunaSearchCV Quickstart](/examples/optuna_search/) - minimal end-to-end walkthrough
        """
    )
    return


if __name__ == "__main__":
    app.run()
