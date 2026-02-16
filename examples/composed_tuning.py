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
        # Composed Tuning with OptunaSearchCV

        ## What You'll Learn

        - How to compose a forecaster with a feature transformer (`LagTransformer`)
        - How to tune nested parameters across the composed pipeline using `__` routing
        - How to use autocorrelation analysis to motivate lag selection
        - How to diagnose forecast quality with residual plots

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
    import optuna
    import polars as pl
    from optuna.distributions import FloatDistribution, IntDistribution
    from sklearn.linear_model import Ridge

    from yohou.datasets import load_sunspots
    from yohou.metrics import MeanAbsoluteError
    from yohou.model_selection import ExpandingWindowSplitter
    from yohou.plotting import (
        plot_autocorrelation,
        plot_cv_results_scatter,
        plot_forecast,
        plot_residual_time_series,
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
        load_sunspots,
        optuna,
        pl,
        plot_autocorrelation,
        plot_cv_results_scatter,
        plot_forecast,
        plot_residual_time_series,
        plot_time_series,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 1. Load and Explore Sunspots

        The Sunspots dataset contains 2820 monthly observations of sunspot activity.
        This long univariate series has strong cyclical patterns (roughly 11-year
        solar cycles), making it ideal for demonstrating lag-based feature engineering.
        We use a subset to keep search times reasonable.
        """
    )
    return


@app.cell
def _(load_sunspots, plot_time_series):
    y_full = load_sunspots()
    y = y_full.tail(500)
    plot_time_series(y, title="Sunspot Activity (Last 500 Months)")
    return y, y_full


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 2. Analyze Autocorrelation Structure

        Before tuning, examine the autocorrelation to understand which lags carry
        the most predictive information. Strong autocorrelation at specific lags
        motivates the choice of lag features for the `LagTransformer`.
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
        ## 3. Compose Forecaster with LagTransformer

        `PointReductionForecaster` converts a time series problem into tabular
        regression. Adding a `LagTransformer` as the `feature_transformer`
        creates lag-based features automatically. We then tune:

        - `estimator__alpha`: Ridge regularization strength
        - `feature_transformer__lag`: number of lag features to include

        Nested parameter names use `__` to route through the composed pipeline.
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
        ## 4. Run Search and Inspect Results

        Fit the search over the composed pipeline. Each trial evaluates a different
        combination of regularization strength and lag count using expanding-window
        cross-validation.
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
        ## 5. Forecast and Diagnose Residuals

        Generate predictions with the best forecaster and evaluate visually.
        The residual plot helps identify systematic errors like bias or
        heteroscedasticity.
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
        ## Key Takeaways

        - **Nested parameter routing** uses `__` to address parameters inside composed pipelines (e.g., `estimator__alpha`, `feature_transformer__lag`)
        - **`LagTransformer`** creates lag-based features automatically, turning time series forecasting into tabular regression
        - **Autocorrelation analysis** helps motivate the range of lags to search over
        - **`ExpandingWindowSplitter`** provides time-respecting cross-validation that avoids data leakage
        - **Residual diagnostics** with `plot_residual_time_series` reveal systematic prediction errors

        ## Next Steps

        - **Multi-metric search**: See multi_metric_search.py to evaluate multiple metrics simultaneously
        - **Search visualization**: See search_visualization.py for Optuna's optimization history and parameter importance plots
        - **Panel data tuning**: See panel_tuning.py to tune forecasters on grouped time series
        """
    )
    return


if __name__ == "__main__":
    app.run()
