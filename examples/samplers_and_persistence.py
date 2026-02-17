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
        # Samplers and Persistence

        This notebook demonstrates how to use different Optuna samplers and
        persistent storage with `OptunaSearchCV`.

        You will learn how to:

        - Use `RandomSampler` and `TPESampler` for different search strategies
        - Enable explicit in-memory storage with the `Storage` wrapper
        - Apply callbacks for early stopping via `MaxTrialsCallback`
        - Continue an existing study across multiple fit calls
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

    from yohou.metrics import MeanAbsoluteError
    from yohou.point import PointReductionForecaster

    from yohou_optuna import Callback, OptunaSearchCV, Sampler, Storage

    return (
        Callback,
        FloatDistribution,
        MeanAbsoluteError,
        OptunaSearchCV,
        PointReductionForecaster,
        Ridge,
        Sampler,
        Storage,
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
        ## 2. Random Sampler

        `RandomSampler` explores the search space uniformly at random.
        Useful as a baseline or when the search space is small.
        """
    )


@app.cell
def _(FloatDistribution, MeanAbsoluteError, OptunaSearchCV, PointReductionForecaster, Ridge, Sampler, X, optuna, y):
    random_search = OptunaSearchCV(
        forecaster=PointReductionForecaster(estimator=Ridge()),
        param_distributions={
            "estimator__alpha": FloatDistribution(0.001, 100.0, log=True),
        },
        scoring=MeanAbsoluteError(),
        sampler=Sampler(sampler=optuna.samplers.RandomSampler, seed=42),
        n_trials=10,
        cv=3,
    )

    random_search.fit(y, X, forecasting_horizon=5)
    return (random_search,)


@app.cell
def _(mo, random_search):
    mo.md(
        f"""
        **Random sampler best score:** `{random_search.best_score_:.4f}`

        **Best params:** `{random_search.best_params_}`
        """
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## 3. TPE Sampler (Default)

        `TPESampler` uses Tree-structured Parzen Estimators for Bayesian
        optimization. This is the default and recommended sampler for
        most hyperparameter search scenarios.
        """
    )


@app.cell
def _(FloatDistribution, MeanAbsoluteError, OptunaSearchCV, PointReductionForecaster, Ridge, Sampler, X, optuna, y):
    tpe_search = OptunaSearchCV(
        forecaster=PointReductionForecaster(estimator=Ridge()),
        param_distributions={
            "estimator__alpha": FloatDistribution(0.001, 100.0, log=True),
        },
        scoring=MeanAbsoluteError(),
        sampler=Sampler(sampler=optuna.samplers.TPESampler, seed=42),
        n_trials=10,
        cv=3,
    )

    tpe_search.fit(y, X, forecasting_horizon=5)
    return (tpe_search,)


@app.cell
def _(mo, tpe_search):
    mo.md(
        f"""
        **TPE sampler best score:** `{tpe_search.best_score_:.4f}`

        **Best params:** `{tpe_search.best_params_}`
        """
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## 4. In-Memory Storage

        By default, Optuna uses in-memory storage. You can make this explicit
        with the `Storage` wrapper for clarity or to later switch to persistent backends.
        """
    )


@app.cell
def _(
    FloatDistribution,
    MeanAbsoluteError,
    OptunaSearchCV,
    PointReductionForecaster,
    Ridge,
    Sampler,
    Storage,
    X,
    optuna,
    y,
):
    stored_search = OptunaSearchCV(
        forecaster=PointReductionForecaster(estimator=Ridge()),
        param_distributions={
            "estimator__alpha": FloatDistribution(0.001, 100.0, log=True),
        },
        scoring=MeanAbsoluteError(),
        sampler=Sampler(sampler=optuna.samplers.TPESampler, seed=42),
        storage=Storage(storage=optuna.storages.InMemoryStorage),
        n_trials=10,
        cv=3,
    )

    stored_search.fit(y, X, forecasting_horizon=5)
    return (stored_search,)


@app.cell
def _(mo, stored_search):
    mo.md(
        f"""
        **Stored search best score:** `{stored_search.best_score_:.4f}`

        **Study has {len(stored_search.study_.trials)} trials.**
        """
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## 5. Callbacks: Early Stopping

        Use `Callback` to wrap Optuna callbacks. Here we use `MaxTrialsCallback`
        to stop the study after a fixed number of trials — useful when
        combined with a high `n_trials` limit and a tight time budget.
        """
    )


@app.cell
def _(
    Callback,
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
    callback_search = OptunaSearchCV(
        forecaster=PointReductionForecaster(estimator=Ridge()),
        param_distributions={
            "estimator__alpha": FloatDistribution(0.001, 100.0, log=True),
        },
        scoring=MeanAbsoluteError(),
        sampler=Sampler(sampler=optuna.samplers.TPESampler, seed=42),
        n_trials=100,  # High limit, but callback stops early
        cv=3,
        callbacks={
            "max_trials": Callback(
                callback=optuna.study.MaxTrialsCallback, n_trials=5
            )
        },
    )

    callback_search.fit(y, X, forecasting_horizon=5)
    return (callback_search,)


@app.cell
def _(callback_search, mo):
    mo.md(
        f"""
        **Completed trials:** `{len(callback_search.trials_)}`
        (Stopped early by MaxTrialsCallback)

        **Best score:** `{callback_search.best_score_:.4f}`
        """
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## 6. Continue an Existing Study

        Pass a previously created Optuna study via the `study` parameter
        in `fit()` to continue optimization from where you left off.
        This lets you incrementally add trials without losing history.
        """
    )


@app.cell
def _(
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
    # First run: 5 trials
    search_1 = OptunaSearchCV(
        forecaster=PointReductionForecaster(estimator=Ridge()),
        param_distributions={
            "estimator__alpha": FloatDistribution(0.001, 100.0, log=True),
        },
        scoring=MeanAbsoluteError(),
        sampler=Sampler(sampler=optuna.samplers.TPESampler, seed=42),
        n_trials=5,
        cv=3,
    )
    search_1.fit(y, X, forecasting_horizon=5)

    # Second run: continue with 5 more trials
    search_2 = OptunaSearchCV(
        forecaster=PointReductionForecaster(estimator=Ridge()),
        param_distributions={
            "estimator__alpha": FloatDistribution(0.001, 100.0, log=True),
        },
        scoring=MeanAbsoluteError(),
        sampler=Sampler(sampler=optuna.samplers.TPESampler, seed=42),
        n_trials=5,
        cv=3,
    )
    search_2.fit(y, X, forecasting_horizon=5, study=search_1.study_)
    return search_1, search_2


@app.cell
def _(mo, search_1, search_2):
    mo.md(
        f"""
        **First run trials:** `{len(search_1.trials_)}`

        **After continuation:** `{len(search_2.trials_)}` total trials

        **Best score (continued):** `{search_2.best_score_:.4f}`
        """
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## Key Takeaways

        - `Sampler` wraps any Optuna sampler (`RandomSampler`, `TPESampler`, etc.) with an optional seed for reproducibility
        - `Storage` wraps Optuna storage backends — use `InMemoryStorage` for ephemeral studies or SQLite for persistence
        - `Callback` wraps Optuna callbacks like `MaxTrialsCallback` for custom trial-level logic
        - Pass `study=previous_search.study_` to `fit()` to continue an existing optimization study
        - All wrappers follow Optuna's API — consult Optuna docs for advanced configurations

        ## Next Steps

        - **[Optuna Search](../optuna_search/)**: Review basics of OptunaSearchCV
        - **[Composed Tuning](../composed_tuning/)**: Tune forecasters with feature transformers
        - **[Search Visualization](../search_visualization/)**: Visualize optimization with Optuna's plotting API
        """
    )


if __name__ == "__main__":
    app.run()
