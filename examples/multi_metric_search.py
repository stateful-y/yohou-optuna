# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "numpy",
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
    "title": "How to Run a Multi-Metric Search",
    "description": "Evaluate MAE, RMSE, and MSE simultaneously and compare how different metrics rank the same trials.",
    "category": "how-to",
    "companion": "pages/how-to/multi-metric-search.md",
}


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # How to Run a Multi-Metric Search

        Pass multiple scorers to `OptunaSearchCV` and choose which metric drives model selection with the `refit` parameter.

        **Prerequisites** - familiarity with [`OptunaSearchCV`](/pages/api/generated/yohou_optuna.search.OptunaSearchCV/) (see [OptunaSearchCV Quickstart](/examples/optuna_search/)).
        """
    )
    return


@app.cell(hide_code=True)
def _():
    import numpy as np
    import optuna
    import polars as pl
    from optuna.distributions import CategoricalDistribution, FloatDistribution
    from sklearn.linear_model import Ridge

    from yohou.datasets import load_ett_m1
    from yohou.metrics import MeanAbsoluteError, MeanSquaredError, RootMeanSquaredError
    from yohou.model_selection import ExpandingWindowSplitter
    from yohou.plotting import (
        plot_correlation_heatmap,
        plot_forecast,
        plot_model_comparison_bar,
        plot_time_series,
    )
    from yohou.point import PointReductionForecaster

    from yohou_optuna import OptunaSearchCV, Sampler

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    return (
        CategoricalDistribution,
        FloatDistribution,
        MeanAbsoluteError,
        MeanSquaredError,
        OptunaSearchCV,
        PointReductionForecaster,
        Ridge,
        RootMeanSquaredError,
        Sampler,
        load_ett_m1,
        np,
        optuna,
        pl,
        plot_correlation_heatmap,
        plot_forecast,
        plot_model_comparison_bar,
        plot_time_series,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 1. Load the Data

        Load a subset of the ETT-M1 dataset (oil temperature + power load features).
        """
    )
    return


@app.cell
def _(load_ett_m1, pl):
    y_full = load_ett_m1()
    y_all = y_full.select(["time", "OT", "HUFL", "HULL"]).head(1000)
    y_all.head()
    return y_all, y_full


@app.cell
def _(plot_time_series, y_all):
    plot_time_series(y_all, title="Electricity Transformer: OT, HUFL, HULL")
    return


@app.cell
def _(plot_correlation_heatmap, y_all):
    plot_correlation_heatmap(y_all, title="Feature Correlation")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 2. Define Multi-Metric Scoring

        Pass a dictionary of scorers to `scoring` and set `refit` to the metric
        that should drive model selection. Set `return_train_score=True` if you
        need overfitting diagnostics.
        """
    )
    return


@app.cell
def _(y_all):
    y_train = y_all.head(800)
    y_test = y_all.tail(200)
    return y_test, y_train


@app.cell
def _(
    CategoricalDistribution,
    FloatDistribution,
    MeanAbsoluteError,
    MeanSquaredError,
    OptunaSearchCV,
    PointReductionForecaster,
    Ridge,
    RootMeanSquaredError,
    Sampler,
    optuna,
):
    search = OptunaSearchCV(
        forecaster=PointReductionForecaster(estimator=Ridge()),
        param_distributions={
            "estimator__alpha": FloatDistribution(0.001, 100.0, log=True),
            "estimator__fit_intercept": CategoricalDistribution([True, False]),
        },
        scoring={
            "mae": MeanAbsoluteError(),
            "rmse": RootMeanSquaredError(),
            "mse": MeanSquaredError(),
        },
        sampler=Sampler(sampler=optuna.samplers.TPESampler, seed=42),
        n_trials=15,
        refit="mae",
        cv=3,
        return_train_score=True,
        verbose=0,
    )
    return (search,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 3. Run the Search

        Fit the search. All three metrics are evaluated per trial; `refit="mae"`
        determines which one selects the best model.
        """
    )
    return


@app.cell
def _(search, y_train):
    search.fit(y_train, forecasting_horizon=24)
    return


@app.cell
def _(mo, search):
    mo.md(
        f"""
        **Best parameters (by MAE):** `{search.best_params_}`

        **Best MAE:** `{search.cv_results_['mean_test_mae'][search.best_index_]:.4f}`

        **Best RMSE:** `{search.cv_results_['mean_test_rmse'][search.best_index_]:.4f}`

        **Best MSE:** `{search.cv_results_['mean_test_mse'][search.best_index_]:.4f}`
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 4. Compare Metrics

        Review per-trial scores and independent rankings for each metric.
        """
    )
    return


@app.cell
def _(pl, search):
    results_df = pl.DataFrame(
        {
            "trial": list(range(len(search.cv_results_["params"]))),
            "alpha": [p.get("estimator__alpha", None) for p in search.cv_results_["params"]],
            "mean_test_mae": search.cv_results_["mean_test_mae"],
            "mean_test_rmse": search.cv_results_["mean_test_rmse"],
            "mean_test_mse": search.cv_results_["mean_test_mse"],
            "rank_mae": search.cv_results_["rank_test_mae"],
            "rank_rmse": search.cv_results_["rank_test_rmse"],
            "mean_train_mae": search.cv_results_["mean_train_mae"],
        }
    ).sort("rank_mae")
    results_df
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Compare the best model's test scores across all three metrics with a bar chart.
        """
    )
    return


@app.cell
def _(np, plot_model_comparison_bar, search):
    best_by_mae_idx = search.best_index_
    best_by_rmse_idx = int(np.argmin(search.cv_results_["rank_test_rmse"]))

    comparison = {
        "Best by MAE": {
            "MAE": abs(search.cv_results_["mean_test_mae"][best_by_mae_idx]),
            "RMSE": abs(search.cv_results_["mean_test_rmse"][best_by_mae_idx]),
            "MSE": abs(search.cv_results_["mean_test_mse"][best_by_mae_idx]),
        },
        "Best by RMSE": {
            "MAE": abs(search.cv_results_["mean_test_mae"][best_by_rmse_idx]),
            "RMSE": abs(search.cv_results_["mean_test_rmse"][best_by_rmse_idx]),
            "MSE": abs(search.cv_results_["mean_test_mse"][best_by_rmse_idx]),
        },
    }
    plot_model_comparison_bar(
        comparison,
        group_by="model",
        title="Best by MAE vs Best by RMSE",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 5. Forecast with the Best Model

        Generate predictions and compare visually against the held-out test set.
        """
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
        title="Multi-Metric Best Forecast (refit=MAE)",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Next Steps

        - [How to Visualize Search Results](/examples/search_visualization/) - Optuna's optimization history and parameter importance plots
        - [How to Tune on Panel Data](/examples/panel_tuning/) - tune forecasters on grouped time series
        - [How to Tune Composed Forecasters](/examples/composed_tuning/) - tune nested pipelines with feature transformers
        """
    )
    return


if __name__ == "__main__":
    app.run()
