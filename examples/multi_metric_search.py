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
        # Multi-Metric Search

        ## What You'll Learn

        - How to define multiple scoring metrics for a single search
        - How `refit` selects which metric drives model selection
        - How to access per-metric results and rankings in `cv_results_`
        - How to visually compare metrics and analyze score per forecast horizon

        ## Prerequisites

        Familiarity with the basics of `OptunaSearchCV` (see optuna_search.py).
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
        ## 1. Load and Explore Multivariate Data

        The ETT-M1 (Electricity Transformer Temperature) dataset contains 15-minute
        measurements of oil temperature and six power load features. We use a subset
        and select a few columns to keep the search fast while still demonstrating
        multivariate forecasting.
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

        Pass a dictionary of scorers to `scoring`. Optuna tracks all metrics per
        trial and selects the best configuration based on the `refit` metric.
        Setting `return_train_score=True` enables overfitting diagnostics.
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
        ## 3. Run Search

        Each trial evaluates all three metrics. The `refit="mae"` setting means
        the final best model is chosen by MAE, even though RMSE and MSE are also
        tracked.
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
        ## 4. Compare Metrics Visually

        The results table shows per-trial scores for all three metrics with
        independent rankings. Different metrics may rank configurations differently.
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
        Compare the best model's test performance across all three metrics
        using a bar chart. This helps assess whether the model performs
        consistently or if some metrics reveal weaknesses.
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
        ## Key Takeaways

        - **Multi-metric scoring** accepts a dictionary of scorers, tracking all metrics per trial
        - **`refit`** determines which metric selects the best model for final refitting
        - **`cv_results_`** provides `mean_test_<name>`, `std_test_<name>`, and `rank_test_<name>` for each metric
        - **`return_train_score=True`** adds training scores for overfitting diagnostics
        - **Different metrics may rank configurations differently** -- use `plot_model_comparison_bar` to compare

        ## Next Steps

        - **Search visualization**: See search_visualization.py for Optuna's optimization history and parameter importance plots
        - **Panel data tuning**: See panel_tuning.py to tune forecasters on grouped time series
        - **Composed tuning**: See composed_tuning.py to tune nested pipelines with feature transformers
        """
    )
    return


if __name__ == "__main__":
    app.run()
