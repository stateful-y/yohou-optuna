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
        # Composed Tuning with OptunaSearchCV

        This notebook demonstrates how to use `OptunaSearchCV` with Yohou's
        composition modules: `ColumnForecaster`, `DecompositionPipeline`,
        and `FeaturePipeline`-based transformers.

        You will learn how to:

        - Combine a feature transformer with a forecaster
        - Tune nested parameters across the composed pipeline
        - Inspect cross-validation results and predict with the best model
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

    from yohou.metrics import MeanAbsoluteError
    from yohou.point import PointReductionForecaster
    from yohou.preprocessing import LagTransformer

    from yohou_optuna import OptunaSearchCV, Sampler

    return (
        CategoricalDistribution,
        FloatDistribution,
        IntDistribution,
        LagTransformer,
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
        ## 1. Create Sample Data

        Generate a time series with trend, seasonality, and noise.
        """
    )


@app.cell(hide_code=True)
def _(datetime, np, pl, timedelta):
    rng = np.random.default_rng(42)
    length = 200
    dates = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(length)]
    trend = np.linspace(0, 10, length)
    seasonal = 3 * np.sin(np.arange(length) * 2 * np.pi / 7)
    noise = rng.normal(0, 0.5, length)
    values = trend + seasonal + noise

    y = pl.DataFrame({"time": dates, "value": values})
    X = pl.DataFrame(
        {
            "time": dates,
            "feature_1": rng.normal(0, 1, length),
        }
    )
    y.head()
    return X, y


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## 2. Tune Forecaster with Feature Transformer

        Search over the estimator's regularization strength
        while using a lag-based feature transformer. Nested parameter names
        like `estimator__alpha` address parameters inside the composed pipeline.
        """
    )


@app.cell
def _(
    FloatDistribution,
    LagTransformer,
    MeanAbsoluteError,
    OptunaSearchCV,
    PointReductionForecaster,
    Ridge,
    Sampler,
    X,
    optuna,
    y,
):
    forecaster = PointReductionForecaster(
        estimator=Ridge(),
        feature_transformer=LagTransformer(lag=[1, 2, 3]),
    )

    search = OptunaSearchCV(
        forecaster=forecaster,
        param_distributions={
            "estimator__alpha": FloatDistribution(0.001, 100.0, log=True),
        },
        scoring=MeanAbsoluteError(),
        sampler=Sampler(sampler=optuna.samplers.TPESampler, seed=42),
        n_trials=10,
        cv=3,
        refit=True,
    )

    search.fit(y, X, forecasting_horizon=5)
    return (search,)


@app.cell
def _(mo, search):
    mo.md(
        f"""
        **Best parameters:** `{search.best_params_}`

        **Best MAE:** `{search.best_score_:.4f}`
        """
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## 3. Inspect CV Results

        View all trial results sorted by rank.
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
            "mean_test_score": search.cv_results_["mean_test_score"],
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

        After refit, the best forecaster is ready for out-of-sample predictions.
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
        ## Key Takeaways

        - Use nested parameter names (e.g., `estimator__alpha`) to tune parameters inside composed pipelines
        - `PointReductionForecaster` accepts a `feature_transformer` for automatic lag-based feature engineering
        - `OptunaSearchCV` handles the full fit–tune–refit lifecycle automatically
        - After fitting, the best forecaster is ready for direct prediction

        ## Next Steps

        - **[Search Visualization](../search_visualization/)**: Visualize optimization history and parameter importances
        - **[Multi-Metric Search](../multi_metric_search/)**: Track multiple metrics simultaneously
        - **[Samplers and Persistence](../samplers_and_persistence/)**: Explore different samplers and study persistence
        """
    )


if __name__ == "__main__":
    app.run()
