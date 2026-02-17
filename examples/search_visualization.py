import marimo

__generated_with = "0.19.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        # Search Visualization with Optuna

        This notebook demonstrates how to visualize hyperparameter search results
        using Optuna's built-in Plotly-based visualization functions.

        You will learn how to:

        - Access the Optuna `study_` object after fitting
        - Plot optimization history, parameter importances, and slice plots
        - Create contour plots for parameter interactions
        """
    )


@app.cell(hide_code=True)
async def _():
    import sys

    if "pyodide" in sys.modules:
        import micropip

        await micropip.install(["scikit-learn", "optuna", "sklearn-optuna"])
    return


@app.cell(hide_code=True)
def _():
    from datetime import datetime, timedelta

    import numpy as np
    import optuna
    import polars as pl
    from optuna.distributions import CategoricalDistribution, FloatDistribution
    from sklearn.linear_model import Ridge

    from yohou.metrics import MeanAbsoluteError
    from yohou.point import PointReductionForecaster

    from yohou_optuna import OptunaSearchCV, Sampler

    return (
        CategoricalDistribution,
        FloatDistribution,
        MeanAbsoluteError,
        OptunaSearchCV,
        PointReductionForecaster,
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
        """
        ## 1. Run the Search

        We search over two parameters to produce interesting visualizations.
        """
    )


@app.cell(hide_code=True)
def _(datetime, np, pl, timedelta):
    rng = np.random.default_rng(42)
    length = 200
    dates = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(length)]
    trend = np.linspace(0, 10, length)
    noise = rng.normal(0, 1, length)
    values = trend + noise

    y = pl.DataFrame({"time": dates, "value": values})
    X = pl.DataFrame(
        {
            "time": dates,
            "feature_1": rng.normal(0, 1, length),
        }
    )
    return X, y


@app.cell
def _(
    CategoricalDistribution,
    FloatDistribution,
    MeanAbsoluteError,
    OptunaSearchCV,
    PointReductionForecaster,
    Ridge,
    Sampler,
    X,
    optuna,
    y,
):
    search = OptunaSearchCV(
        forecaster=PointReductionForecaster(estimator=Ridge()),
        param_distributions={
            "estimator__alpha": FloatDistribution(0.001, 100.0, log=True),
            "estimator__fit_intercept": CategoricalDistribution([True, False]),
        },
        scoring=MeanAbsoluteError(),
        sampler=Sampler(sampler=optuna.samplers.TPESampler, seed=42),
        n_trials=20,
        cv=3,
        refit=True,
    )

    search.fit(y, X, forecasting_horizon=5)
    return (search,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## 2. Optimization History

        Shows the objective value across trials, with the running best highlighted.
        """
    )


@app.cell
def _(optuna, search):
    fig_history = optuna.visualization.plot_optimization_history(search.study_)
    fig_history
    return (fig_history,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## 3. Parameter Importances

        Displays how much each hyperparameter contributes to score variation.
        """
    )


@app.cell
def _(optuna, search):
    fig_importance = optuna.visualization.plot_param_importances(search.study_)
    fig_importance
    return (fig_importance,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## 4. Parameter Slice Plot

        Shows the relationship between individual parameter values and the objective.
        """
    )


@app.cell
def _(optuna, search):
    fig_slice = optuna.visualization.plot_slice(search.study_)
    fig_slice
    return (fig_slice,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## 5. Contour Plot

        Visualizes the interaction between two parameters as a 2D surface.
        """
    )


@app.cell
def _(optuna, search):
    fig_contour = optuna.visualization.plot_contour(
        search.study_,
        params=["estimator__alpha", "estimator__fit_intercept"],
    )
    fig_contour
    return (fig_contour,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## Key Takeaways

        - After fitting, `search.study_` exposes the full Optuna study for visualization
        - `plot_optimization_history` shows convergence across trials
        - `plot_param_importances` reveals which hyperparameters matter most
        - `plot_slice` and `plot_contour` help understand individual and joint parameter effects
        - All visualization functions return Plotly figures for interactive exploration

        ## Next Steps

        - **[Multi-Metric Search](../multi_metric_search/)**: Track multiple metrics and compare rankings
        - **[Samplers and Persistence](../samplers_and_persistence/)**: Custom samplers, callbacks, and study persistence
        - **[Optuna Search](../optuna_search/)**: Review basics of OptunaSearchCV
        """
    )


if __name__ == "__main__":
    app.run()
