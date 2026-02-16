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
        # Optuna Hyperparameter Search for Yohou

        ## What You'll Learn

        - How to load a real dataset from `yohou.datasets`
        - How to define Optuna search distributions for forecaster parameters
        - How to run `OptunaSearchCV` for Bayesian hyperparameter optimization
        - How to inspect cross-validation results and visualize them with `yohou.plotting`
        - How to generate and visualize forecasts with the best model

        ## Prerequisites

        Basic familiarity with scikit-learn's fit/predict API and time series forecasting concepts.
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
    from optuna.distributions import CategoricalDistribution, FloatDistribution
    from sklearn.linear_model import Ridge

    from yohou.datasets import load_air_passengers
    from yohou.metrics import MeanAbsoluteError
    from yohou.plotting import plot_cv_results_scatter, plot_forecast, plot_time_series
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
        load_air_passengers,
        optuna,
        pl,
        plot_cv_results_scatter,
        plot_forecast,
        plot_time_series,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 1. Load and Explore the Data

        We use the classic Air Passengers dataset: 144 monthly observations of
        international airline passenger counts from 1949 to 1960. This univariate
        series exhibits both trend and seasonality, making it a good benchmark for
        forecasting.
        """
    )
    return


@app.cell
def _(load_air_passengers, plot_time_series):
    y = load_air_passengers()
    plot_time_series(y, title="Air Passengers (1949-1960)")
    return (y,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Split the data into training (first 120 months) and test (last 24 months) sets.
        The test set will be used later to evaluate the best forecaster.
        """
    )
    return


@app.cell
def _(y):
    y_train = y.head(120)
    y_test = y.tail(24)
    return y_test, y_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 2. Define Forecaster and Search Space

        We use a `PointReductionForecaster` with a Ridge regression estimator.
        The search space covers the regularization strength (`alpha`) and whether
        to fit an intercept. Optuna's TPE sampler performs Bayesian optimization
        over these distributions.
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
):
    param_distributions = {
        "estimator__alpha": FloatDistribution(0.001, 100.0, log=True),
        "estimator__fit_intercept": CategoricalDistribution([True, False]),
    }

    search = OptunaSearchCV(
        forecaster=PointReductionForecaster(estimator=Ridge()),
        param_distributions=param_distributions,
        scoring=MeanAbsoluteError(),
        sampler=Sampler(sampler=optuna.samplers.TPESampler, seed=42),
        n_trials=20,
        cv=3,
        refit=True,
        verbose=0,
    )
    return (search,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 3. Run OptunaSearchCV

        Fit the search on the training data. `OptunaSearchCV` runs cross-validated
        evaluation for each trial and uses the TPE sampler to guide the search
        toward promising regions of the hyperparameter space.
        """
    )
    return


@app.cell
def _(search, y_train):
    search.fit(y_train, forecasting_horizon=12)
    return


@app.cell
def _(mo, search):
    mo.md(
        f"""
        **Best parameters:** `{search.best_params_}`

        **Best score (MAE):** `{search.best_score_:.4f}`

        **Number of trials:** `{len(search.trials_)}`
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 4. Inspect Cross-Validation Results

        The `cv_results_` dictionary contains per-trial scores, parameters, and
        rankings. We can visualize how the regularization strength affects the
        MAE score using `plot_cv_results_scatter`.
        """
    )
    return


@app.cell
def _(pl, search):
    results_df = pl.DataFrame(
        {
            "trial": list(range(len(search.cv_results_["params"]))),
            "alpha": [p.get("estimator__alpha", None) for p in search.cv_results_["params"]],
            "fit_intercept": [p.get("estimator__fit_intercept", None) for p in search.cv_results_["params"]],
            "mean_test_score": search.cv_results_["mean_test_score"],
            "std_test_score": search.cv_results_["std_test_score"],
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 5. Forecast and Evaluate

        After fitting, `OptunaSearchCV` refits the best forecaster on the full
        training data. Use it directly to generate forecasts and compare against
        the held-out test set.
        """
    )
    return


@app.cell
def _(search):
    y_pred = search.predict(forecasting_horizon=12)
    y_pred
    return (y_pred,)


@app.cell
def _(plot_forecast, y_pred, y_test, y_train):
    plot_forecast(
        y_test,
        y_pred,
        y_train=y_train,
        n_history=36,
        title="Best Forecaster: Predicted vs Actual",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Key Takeaways

        - **OptunaSearchCV** wraps Optuna's Bayesian optimization for yohou forecasters with a familiar scikit-learn-style API
        - **Search distributions** like `FloatDistribution` and `CategoricalDistribution` define the hyperparameter space
        - **`plot_cv_results_scatter`** visualizes how parameter values relate to cross-validation scores
        - **`plot_forecast`** overlays predicted values on actuals for quick visual evaluation
        - **`best_params_`, `best_score_`, `cv_results_`** provide full access to search results after fitting

        ## Next Steps

        - **Composed tuning**: See composed_tuning.py to tune forecasters with feature transformers like `LagTransformer`
        - **Multi-metric search**: See multi_metric_search.py to track multiple metrics (MAE, RMSE, MSE) simultaneously
        - **Search visualization**: See search_visualization.py for Optuna's built-in optimization plots
        """
    )
    return


if __name__ == "__main__":
    app.run()
