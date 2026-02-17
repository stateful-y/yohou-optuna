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
        # Multi-Metric Search

        This notebook demonstrates advanced multi-metric hyperparameter search
        with `OptunaSearchCV`. When you use a dictionary of scorers, Optuna
        tracks all metrics per trial and selects the best configuration based
        on the `refit` metric.

        You will learn how to:

        - Define multiple scoring metrics for a single search
        - Use `refit` to select which metric drives model selection
        - Access per-metric results and rankings in `cv_results_`
        - Compare rankings across different metrics
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
    from optuna.distributions import FloatDistribution
    from sklearn.linear_model import Ridge

    from yohou.metrics import MeanAbsoluteError, MeanSquaredError, RootMeanSquaredError
    from yohou.point import PointReductionForecaster

    from yohou_optuna import OptunaSearchCV, Sampler

    return (
        FloatDistribution,
        MeanAbsoluteError,
        MeanSquaredError,
        OptunaSearchCV,
        PointReductionForecaster,
        Ridge,
        RootMeanSquaredError,
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
        ## 1. Create Sample Data
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## 2. Search with Three Metrics

        Track MAE, RMSE, and MSE simultaneously. Use `refit="mae"` to select
        the best model by MAE while still recording the other metrics.
        """
    )


@app.cell
def _(
    FloatDistribution,
    MeanAbsoluteError,
    MeanSquaredError,
    OptunaSearchCV,
    PointReductionForecaster,
    Ridge,
    RootMeanSquaredError,
    Sampler,
    X,
    optuna,
    y,
):
    search = OptunaSearchCV(
        forecaster=PointReductionForecaster(estimator=Ridge()),
        param_distributions={
            "estimator__alpha": FloatDistribution(0.001, 100.0, log=True),
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
    )

    search.fit(y, X, forecasting_horizon=5)
    return (search,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## 3. Results Overview

        All three metrics are available in `cv_results_` with per-metric
        rankings computed independently.
        """
    )


@app.cell
def _(mo, search):
    mo.md(
        f"""
        **Best parameters:** `{search.best_params_}`

        **Best MAE:** `{search.cv_results_['mean_test_mae'][search.best_index_]:.4f}`

        **Best RMSE:** `{search.cv_results_['mean_test_rmse'][search.best_index_]:.4f}`

        **Best MSE:** `{search.cv_results_['mean_test_mse'][search.best_index_]:.4f}`
        """
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## 4. Detailed CV Results Table

        Examine per-trial scores for all metrics alongside train scores.
        """
    )


@app.cell
def _(pl, search):
    results_df = pl.DataFrame(
        {
            "trial": list(range(len(search.cv_results_["params"]))),
            "alpha": [
                p.get("estimator__alpha", None)
                for p in search.cv_results_["params"]
            ],
            "mean_test_mae": search.cv_results_["mean_test_mae"],
            "mean_test_rmse": search.cv_results_["mean_test_rmse"],
            "mean_test_mse": search.cv_results_["mean_test_mse"],
            "rank_mae": search.cv_results_["rank_test_mae"],
            "rank_rmse": search.cv_results_["rank_test_rmse"],
            "rank_mse": search.cv_results_["rank_test_mse"],
            "mean_train_mae": search.cv_results_["mean_train_mae"],
        }
    ).sort("rank_mae")
    results_df
    return (results_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## 5. Compare Rankings

        Different metrics may produce different rankings. Check whether the
        best model by MAE is also best by RMSE.
        """
    )


@app.cell
def _(mo, np, search):
    best_mae_idx = np.argmin(search.cv_results_["rank_test_mae"])
    best_rmse_idx = np.argmin(search.cv_results_["rank_test_rmse"])

    if best_mae_idx == best_rmse_idx:
        msg = "The same configuration ranks first for both MAE and RMSE."
    else:
        msg = (
            f"Different configurations rank first: "
            f"MAE best at trial {best_mae_idx}, RMSE best at trial {best_rmse_idx}."
        )
    mo.md(msg)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## Key Takeaways

        - Pass a dictionary of scorers to `scoring` for multi-metric search
        - The `refit` parameter selects which metric determines the best model
        - `cv_results_` contains `mean_test_<name>`, `std_test_<name>`, and `rank_test_<name>` for each metric
        - Use `return_train_score=True` to also collect training scores (useful for overfitting diagnostics)
        - Different metrics may rank configurations differently — always check consistency

        ## Next Steps

        - **[Search Visualization](../search_visualization/)**: Visualize optimization with Optuna's plotting API
        - **[Samplers and Persistence](../samplers_and_persistence/)**: Custom samplers, callbacks, and study persistence
        - **[Composed Tuning](../composed_tuning/)**: Tune forecasters with feature transformers
        """
    )


if __name__ == "__main__":
    app.run()
