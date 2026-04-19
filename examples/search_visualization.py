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
    "title": "Search Visualization",
    "description": "Combine Optuna's optimization plots with yohou's forecast diagnostics for comprehensive analysis.",
    "category": "Advanced Topics",
}


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Search Visualization with Optuna

        ## What You'll Learn

        - How to access the Optuna `study_` object after fitting [`OptunaSearchCV`](/pages/api/generated/yohou_optuna.search.OptunaSearchCV/)
        - How to plot optimization history, parameter importances, and contour plots with Optuna
        - How to use `yohou.plotting` for CV result scatter plots, forecasts, and residual diagnostics
        - How to combine Optuna and yohou visualizations for comprehensive analysis

        ## Prerequisites

        Familiarity with the basics of [`OptunaSearchCV`](/pages/api/generated/yohou_optuna.search.OptunaSearchCV/) (see [`optuna_search.py`](/examples/optuna_search/)).
        """
    )
    return


@app.cell(hide_code=True)
def _():
    import optuna
    import polars as pl
    from optuna.distributions import CategoricalDistribution, FloatDistribution
    from sklearn.linear_model import Ridge

    from yohou.datasets import fetch_electricity_demand
    from yohou.metrics import MeanAbsoluteError
    from yohou.plotting import (
        plot_cv_results_scatter,
        plot_forecast,
        plot_residuals,
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
        fetch_electricity_demand,
        optuna,
        pl,
        plot_cv_results_scatter,
        plot_forecast,
        plot_residuals,
        plot_time_series,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 1. Load and Explore the Data

        The Australian Electricity Demand dataset contains 30-minute measurements
        of electricity demand across multiple states. We use a subset of Victoria's
        demand column to keep the search fast.
        """
    )
    return


@app.cell
def _(fetch_electricity_demand, pl):
    y_full = fetch_electricity_demand().frame
    y_all = y_full.select(["time", "vic__demand"]).rename({"vic__demand": "demand"}).head(500)
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

        We search over two parameters to produce interesting visualizations:
        the regularization strength and the intercept setting. Using 25 trials
        generates enough data for meaningful optimization plots.
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

        After fitting, `search.study_` exposes the underlying Optuna study.
        Optuna provides built-in Plotly visualizations for analyzing the search.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Optimization History** shows the objective value across trials with a running best line.
        This reveals whether the sampler is converging.
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
        **Parameter Importances** show how much each hyperparameter contributes to
        objective variation. This helps decide which parameters to keep tuning.
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
        **Slice Plots** show the relationship between individual parameter values
        and the objective. Each point is a trial.
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
        **Contour Plot** visualizes the interaction between two parameters as a 2D
        surface, revealing joint effects not visible in slice plots.
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

        Yohou's plotting module complements Optuna's study-level visualizations
        with forecast-level diagnostics. These show how well the best model
        actually performs on the data.
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
def _(plot_residuals, y_pred, y_test):
    plot_residuals(
        y_pred,
        y_test,
        title="Forecast Residuals",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Key Takeaways

        - **`search.study_`** exposes the full Optuna study for native visualization
        - **`plot_optimization_history`** reveals sampler convergence across trials
        - **`plot_param_importances`** identifies which hyperparameters matter most
        - **`plot_slice` and `plot_contour`** show individual and joint parameter effects
        - **`plot_cv_results_scatter`** from yohou visualizes score vs parameter values
        - **`plot_forecast` and `plot_residuals`** provide forecast-level diagnostics

        ## Next Steps

        - **Multi-metric search**: See [`multi_metric_search.py`](/examples/multi_metric_search/) to track multiple metrics and compare rankings
        - **Panel data tuning**: See [`panel_tuning.py`](/examples/panel_tuning/) for grouped time series optimization
        - **Quickstart**: See [`optuna_search.py`](/examples/optuna_search/) for a minimal end-to-end walkthrough
        """
    )
    return


if __name__ == "__main__":
    app.run()
