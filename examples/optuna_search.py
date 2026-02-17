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
        # Optuna Hyperparameter Search for Yohou

        This notebook demonstrates how to use `OptunaSearchCV` from `yohou-optuna`
        for Bayesian hyperparameter optimization of time series forecasters.

        You will learn how to:

        - Define an Optuna search space with distributions
        - Run a basic single-metric search
        - Inspect cross-validation results and best parameters
        - Use multi-metric scoring with a refit strategy
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
    from optuna.distributions import CategoricalDistribution, FloatDistribution, IntDistribution
    from sklearn.linear_model import Ridge
    from yohou.metrics import MeanAbsoluteError, RootMeanSquaredError
    from yohou.point import PointReductionForecaster

    from yohou_optuna import OptunaSearchCV, Sampler

    return (
        CategoricalDistribution,
        FloatDistribution,
        IntDistribution,
        MeanAbsoluteError,
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

        Generate a simple time series with trend and noise for demonstration.
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
            "feature_2": np.sin(np.arange(length) * 2 * np.pi / 7),
        }
    )
    y.head()
    return X, y


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## 2. Basic OptunaSearchCV

        Search over the regularization strength of the underlying estimator.
        Optuna uses TPESampler by default for intelligent Bayesian exploration.
        """
    )


@app.cell
def _(FloatDistribution, MeanAbsoluteError, OptunaSearchCV, PointReductionForecaster, Ridge, Sampler, X, optuna, y):
    param_distributions = {
        "estimator__alpha": FloatDistribution(0.001, 100.0, log=True),
    }

    search = OptunaSearchCV(
        forecaster=PointReductionForecaster(estimator=Ridge()),
        param_distributions=param_distributions,
        scoring=MeanAbsoluteError(),
        sampler=Sampler(sampler=optuna.samplers.TPESampler, seed=42),
        n_trials=10,
        cv=3,
        refit=True,
    )

    search.fit(y, X, forecasting_horizon=5)
    return (search,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## 3. Inspect Results

        View the best parameters, best score, and full cross-validation results.
        """
    )


@app.cell
def _(mo, search):
    mo.md(
        f"""
        **Best parameters:** `{search.best_params_}`

        **Best score (MAE):** `{search.best_score_:.4f}`

        **Number of trials:** `{len(search.trials_)}`
        """
    )


@app.cell
def _(pl, search):
    results_df = pl.DataFrame(
        {
            "trial": list(range(len(search.cv_results_["params"]))),
            "alpha": [p.get("estimator__alpha", None) for p in search.cv_results_["params"]],
            "mean_test_score": search.cv_results_["mean_test_score"],
            "std_test_score": search.cv_results_["std_test_score"],
            "rank": search.cv_results_["rank_test_score"],
        }
    ).sort("rank")
    results_df
    return (results_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## 4. Predict with Best Forecaster

        After fitting, `OptunaSearchCV` refits the best forecaster
        on the full dataset. Use it directly for predictions.
        """
    )


@app.cell
def _(search):
    y_pred = search.predict(forecasting_horizon=5)
    y_pred
    return (y_pred,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## 5. Multi-Metric Search

        Use multiple scoring metrics simultaneously. Specify `refit`
        to choose which metric selects the best model.
        """
    )


@app.cell
def _(
    CategoricalDistribution,
    FloatDistribution,
    MeanAbsoluteError,
    OptunaSearchCV,
    PointReductionForecaster,
    Ridge,
    RootMeanSquaredError,
    Sampler,
    X,
    optuna,
    y,
):
    multi_search = OptunaSearchCV(
        forecaster=PointReductionForecaster(estimator=Ridge()),
        param_distributions={
            "estimator__alpha": FloatDistribution(0.001, 100.0, log=True),
            "estimator__fit_intercept": CategoricalDistribution([True, False]),
        },
        scoring={
            "mae": MeanAbsoluteError(),
            "rmse": RootMeanSquaredError(),
        },
        sampler=Sampler(sampler=optuna.samplers.TPESampler, seed=42),
        n_trials=10,
        refit="mae",
        cv=3,
    )

    multi_search.fit(y, X, forecasting_horizon=5)
    return (multi_search,)


@app.cell
def _(mo, multi_search):
    mo.md(
        f"""
        **Multi-metric best params:** `{multi_search.best_params_}`

        **Best MAE:** `{multi_search.cv_results_['mean_test_mae'][multi_search.best_index_]:.4f}`

        **Best RMSE:** `{multi_search.cv_results_['mean_test_rmse'][multi_search.best_index_]:.4f}`
        """
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## Key Takeaways

        - `OptunaSearchCV` wraps Optuna's Bayesian optimization for yohou forecasters
        - Use `FloatDistribution`, `IntDistribution`, `CategoricalDistribution` to define search spaces
        - Pass a `Sampler` wrapper for reproducible searches with a fixed seed
        - Multi-metric scoring requires `refit` to specify which metric selects the best model
        - After fitting, `best_params_`, `best_score_`, and `cv_results_` are available for inspection

        ## Next Steps

        - **[Composed Tuning](../composed_tuning/)**: Tune forecasters with feature transformers
        - **[Search Visualization](../search_visualization/)**: Visualize optimization with Optuna's plotting API
        - **[Multi-Metric Search](../multi_metric_search/)**: Deep dive into multi-metric strategies
        - **[Samplers and Persistence](../samplers_and_persistence/)**: Custom samplers, callbacks, and study persistence
        """
    )


if __name__ == "__main__":
    app.run()
