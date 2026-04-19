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
    "title": "How to Tune Composed Forecasters",
    "description": "Tune nested parameters across a Ridge regressor and LagTransformer pipeline using double-underscore routing.",
    "category": "how-to",
    "companion": "pages/how-to/composed-forecasters.md",
}


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # How to Tune Composed Forecasters

        Tune nested parameters across a `PointReductionForecaster` + `LagTransformer` pipeline using `__` parameter routing.

        **Prerequisites**: familiarity with [`OptunaSearchCV`](/pages/api/generated/yohou_optuna.search.OptunaSearchCV/) (see [OptunaSearchCV Quickstart](/examples/optuna_search/)).
        """
    )
    return


@app.cell(hide_code=True)
def _():
    import optuna
    import polars as pl
    from optuna.distributions import FloatDistribution, IntDistribution
    from sklearn.linear_model import Ridge

    from yohou.datasets import fetch_sunspot
    from yohou.metrics import MeanAbsoluteError
    from yohou.model_selection import ExpandingWindowSplitter
    from yohou.plotting import (
        plot_autocorrelation,
        plot_cv_results_scatter,
        plot_forecast,
        plot_residuals,
        plot_time_series,
    )
    from yohou.point import PointReductionForecaster
    from yohou.preprocessing import LagTransformer

    from yohou_optuna import OptunaSearchCV, Sampler

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    return (
        ExpandingWindowSplitter,
        FloatDistribution,
        IntDistribution,
        LagTransformer,
        MeanAbsoluteError,
        OptunaSearchCV,
        PointReductionForecaster,
        Ridge,
        Sampler,
        fetch_sunspot,
        optuna,
        pl,
        plot_autocorrelation,
        plot_cv_results_scatter,
        plot_forecast,
        plot_residuals,
        plot_time_series,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 1. Load the Sunspots Data

        Load a subset of the Sunspots dataset (2820 monthly observations) and split into train/test.
        """
    )
    return


@app.cell
def _(fetch_sunspot, plot_time_series):
    y_full = fetch_sunspot().frame
    y = y_full.tail(500)
    plot_time_series(y, title="Sunspot Activity (Last 500 Months)")
    return y, y_full


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 2. Check the Autocorrelation

        Plot the autocorrelation to identify which lags to include in the search range.
        """
    )
    return


@app.cell
def _(plot_autocorrelation, y):
    plot_autocorrelation(y, max_lags=40, title="Autocorrelation of Sunspot Activity")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 3. Compose the Forecaster and Define the Search Space

        Add a `LagTransformer` as the `feature_transformer` and define distributions
        for both `estimator__alpha` and `feature_transformer__lag`. The `__` routing
        addresses parameters inside the composed pipeline.
        """
    )
    return


@app.cell
def _(y):
    y_train = y.head(400)
    y_test = y.tail(24)
    return y_test, y_train


@app.cell
def _(
    ExpandingWindowSplitter,
    FloatDistribution,
    IntDistribution,
    LagTransformer,
    MeanAbsoluteError,
    OptunaSearchCV,
    PointReductionForecaster,
    Ridge,
    Sampler,
    optuna,
):
    forecaster = PointReductionForecaster(
        estimator=Ridge(),
        feature_transformer=LagTransformer(lag=6),
    )

    search = OptunaSearchCV(
        forecaster=forecaster,
        param_distributions={
            "estimator__alpha": FloatDistribution(0.001, 100.0, log=True),
            "feature_transformer__lag": IntDistribution(3, 24),
        },
        scoring=MeanAbsoluteError(),
        sampler=Sampler(sampler=optuna.samplers.TPESampler, seed=42),
        n_trials=20,
        cv=ExpandingWindowSplitter(n_splits=3, test_size=24),
        refit=True,
        verbose=0,
    )
    return (search,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 4. Run the Search and Inspect Results

        Fit the search and review per-trial scores.
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
        **Best parameters:** `{search.best_params_}`

        **Best MAE:** `{search.best_score_:.4f}`
        """
    )
    return


@app.cell
def _(pl, search):
    results_df = pl.DataFrame(
        {
            "trial": list(range(len(search.cv_results_["params"]))),
            "alpha": [p.get("estimator__alpha", None) for p in search.cv_results_["params"]],
            "lag": [p.get("feature_transformer__lag", None) for p in search.cv_results_["params"]],
            "mean_test_score": search.cv_results_["mean_test_score"],
            "rank": search.cv_results_["rank_test_score"],
        }
    ).sort("rank")
    results_df
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
def _(plot_cv_results_scatter, search):
    plot_cv_results_scatter(
        search.cv_results_,
        param_name="feature_transformer__lag",
        higher_is_better=False,
        title="CV Score vs Number of Lags",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 5. Forecast and Check Residuals

        Generate predictions with the best forecaster and check the residuals.
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
        n_history=60,
        title="Sunspot Forecast: Best Composed Model",
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

        - **Nested parameter routing** uses `__` to address parameters inside composed pipelines (e.g., `estimator__alpha`, `feature_transformer__lag`)
        - **`LagTransformer`** creates lag-based features automatically, turning time series forecasting into tabular regression
        - **Autocorrelation analysis** helps motivate the range of lags to search over
        - **`ExpandingWindowSplitter`** provides time-respecting cross-validation that avoids data leakage
        - **Residual diagnostics** with `plot_residuals` reveal systematic prediction errors

        ## Next Steps

        - [How to Run a Multi-Metric Search](/examples/multi_metric_search/): evaluate multiple metrics simultaneously
        - [How to Visualize Search Results](/examples/search_visualization/) - Optuna's optimization history and parameter importance plots
        - [How to Tune on Panel Data](/examples/panel_tuning/): tune forecasters on grouped time series
        """
    )
    return


if __name__ == "__main__":
    app.run()
