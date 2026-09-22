"""Objective function for Optuna hyperparameter optimization in Yohou."""

from __future__ import annotations

import logging
import numbers
import time
from typing import Any, cast

import numpy as np
import optuna
import polars as pl
from sklearn.base import clone
from sklearn.utils.metaestimators import _safe_split
from sklearn.utils.validation import _check_method_params
from yohou.base import BaseForecaster
from yohou.metrics.base import BaseScorer
from yohou.model_selection.utils import (
    _evaluate_candidate_shared_rounds,
    _MultimetricScorer,
    _predict,
    _score,
    _score_train_window,
    _split_X_forecast,
    _train_window_predictions,
)

logger = logging.getLogger(__name__)


class _SharedRoundConfigurationError(ValueError):
    """A trial's parameters that yohou refuses for ``validation="cv"``, raised before any fold is fitted.

    Distinct from a fold failure, which ``error_score`` governs: it stops the
    search, as the same configuration stops yohou's ``GridSearchCV``.
    """


class _Objective:
    """Objective function for Optuna trials in OptunaSearchCV.

    This class encapsulates the logic for evaluating hyperparameter
    configurations during Optuna optimization. It handles parameter
    suggestion, cross-validation using Yohou's time series scoring,
    and error handling.

    Parameters
    ----------
    forecaster : BaseForecaster
        Base forecaster to evaluate.
    param_distributions : dict[str, optuna.distributions.BaseDistribution]
        Dictionary mapping parameter names to Optuna distributions.
    y : pl.DataFrame
        Target time series with a ``"time"`` column.
    X_actual : pl.DataFrame or None
        Actual observation features with a ``"time"`` column.
    X_future : pl.DataFrame or None
        Known future features with a ``"time"`` column.
    X_forecast : pl.DataFrame or None
        External forecasts with ``"vintage_time"`` and ``"time"``
        columns.
    forecasting_horizon : int
        Number of steps ahead to forecast.
    cv : BaseSplitter
        Cross-validation splitter.
    scorers : BaseScorer or _MultimetricScorer
        Scoring functions.
    fit_params : dict
        Additional parameters passed to ``forecaster.fit()``.
    predict_func_params : dict
        Additional parameters passed to ``forecaster.predict()`` or ``forecaster.predict_interval()``.
    score_params : dict
        Additional parameters passed to scorer.
    split_params : dict
        Additional parameters passed to ``cv.split()``.
    verbose : int, default=0
        Verbosity level.
    return_train_score : bool, default=False
        Whether to include training scores.
    error_score : numeric or 'raise', default=np.nan
        Value a failed fold contributes to the trial's fold mean, or
        ``'raise'`` to propagate exceptions.  With the NaN default, one
        failed fold makes the aggregate NaN and the trial carries the
        sentinel objective.
    multimetric : bool, default=False
        Whether multiple metrics are being optimized.
    refit : bool or str, default=True
        Primary metric name for multi-metric optimization.
    coverage_rates : list of float or None, default=None
        Coverage rates for interval forecasters.  Passed to
        ``forecaster.fit()``.
    validation : {"cv"} or None, default=None
        ``"cv"`` evaluates each trial through yohou's shared-round early
        stopping (``_evaluate_candidate_shared_rounds``): every fold's test
        window is its evaluation set, one round per estimator is chosen from
        the fold-average stopping curve, and every fold is scored cut to it.
        ``None`` fits and scores each fold as configured.
    early_stopping_adapter : BaseEarlyStoppingAdapter or None, default=None
        Adapter for ``validation="cv"``, or ``None`` for yohou's built-in one.

    Notes
    -----
    The trial evaluation flow is:

    1. Suggest parameters from distributions using ``trial._suggest()``.
    2. Store parameters as trial user attributes (``param_{name}``).
    3. Clone forecaster with suggested parameters.
    4. Run cross-validation across all splits.
    5. Store per-split scores, timing, and aggregated statistics.
    6. Return the mean test score (or primary metric in multi-metric mode).

    If an error occurs during evaluation, behavior depends on
    ``error_score``.  If ``'raise'``, the exception propagates.  Otherwise
    the failed fold contributes ``error_score`` to a plain mean over every
    fold, exactly as sklearn does: a numeric value participates at face
    value, and the NaN default makes the aggregate NaN, in which case the
    objective returns ``-inf``.  A trial is never scored on only the folds
    it survived; the folds are shared across trials, so a subset score
    would be built from the trial's easiest evidence.

    A fold failure that is absorbed is recorded on the trial: the first
    exception's type and message under ``exception`` and ``exception_type``
    (the same keys the trial-level handler uses), and the failed split
    indices under ``failed_splits``.  One warning is emitted per failing
    trial, so a universal failure stays readable at any trial count.

    """

    def __init__(
        self,
        forecaster: BaseForecaster,
        param_distributions: dict[str, Any],
        y: pl.DataFrame,
        X_actual: pl.DataFrame | None,
        X_future: pl.DataFrame | None,
        X_forecast: pl.DataFrame | None,
        forecasting_horizon: int,
        cv: Any,
        scorers: BaseScorer | _MultimetricScorer,
        fit_params: dict[str, Any],
        predict_func_params: dict[str, Any],
        score_params: dict[str, Any],
        split_params: dict[str, Any],
        *,
        verbose: int = 0,
        return_train_score: bool = False,
        error_score: float | str = np.nan,
        multimetric: bool = False,
        refit: bool | str = True,
        coverage_rates: list[float] | None = None,
        validation: str | None = None,
        early_stopping_adapter: Any = None,
    ) -> None:
        self.forecaster = forecaster
        self.param_distributions = param_distributions
        self.y = y
        self.X_actual = X_actual
        self.X_future = X_future
        self.X_forecast = X_forecast
        self.forecasting_horizon = forecasting_horizon
        self.cv = cv
        self.scorers = scorers
        self.fit_params = fit_params
        self.predict_func_params = predict_func_params
        self.score_params = score_params
        self.split_params = split_params
        self.verbose = verbose
        self.return_train_score = return_train_score
        self.error_score = error_score
        self.multimetric = multimetric
        self.refit = refit
        self.coverage_rates = coverage_rates
        self.validation = validation
        self.early_stopping_adapter = early_stopping_adapter

    def __call__(self, trial: optuna.trial.Trial) -> float:
        """Evaluate a single trial.

        Parameters
        ----------
        trial : optuna.trial.Trial
            Optuna trial object for suggesting parameters.

        Returns
        -------
        float
            Optimization objective value (score to maximize).  Returns
            ``-inf`` if the trial fails and ``error_score`` is not
            ``'raise'``.

        Raises
        ------
        Exception
            If ``error_score='raise'`` and an error occurs during
            forecaster fitting or scoring.

        """
        # Suggest parameters
        study_params = self._suggest_parameters(trial)

        # Store parameters as user attributes
        self._store_parameters(trial, study_params)

        # Optuna logs a trial only at completion, so a process that dies mid-trial
        # leaves no record of which trials were in flight; this line is that record.
        logger.info("Trial %d started with parameters: %s", trial.number, study_params)

        try:
            # Run cross-validation
            self._run_cross_validation(trial, study_params)

            # Return the primary metric
            if self.multimetric:
                return self._get_primary_metric(trial)
            else:
                mean_score = trial.user_attrs.get("mean_test_score", float("nan"))
                if np.isnan(mean_score):
                    return float("-inf")
                return mean_score

        except _SharedRoundConfigurationError:
            # A candidate yohou refuses for validation="cv" stops the search, as it
            # stops GridSearchCV, instead of being recorded as a failed trial.
            raise
        except Exception as e:
            return self._handle_error(trial, e)

    def _suggest_parameters(self, trial: optuna.trial.Trial) -> dict[str, Any]:
        """Suggest parameters from distributions.

        Parameters
        ----------
        trial : optuna.trial.Trial
            Optuna trial for parameter suggestion.

        Returns
        -------
        dict
            Suggested parameter values.

        """
        study_params = {}
        for param_name, distribution in self.param_distributions.items():
            study_params[param_name] = trial._suggest(param_name, distribution)
        return study_params

    def _store_parameters(self, trial: optuna.trial.Trial, params: dict[str, Any]) -> None:
        """Store parameters as trial user attributes.

        Parameters
        ----------
        trial : optuna.trial.Trial
            Optuna trial to store attributes on.
        params : dict
            Parameter values to store.

        """
        for param_name, param_value in params.items():
            trial.set_user_attr(f"param_{param_name}", param_value)

    def _run_cross_validation(self, trial: optuna.trial.Trial, params: dict[str, Any]) -> None:
        """Run cross-validation with given parameters and store results on trial.

        Parameters
        ----------
        trial : optuna.trial.Trial
            Optuna trial for storing results.
        params : dict
            Parameter settings for the forecaster.

        """
        cloned_forecaster = clone(self.forecaster)
        cloned_forecaster.set_params(**params)

        splits = list(self.cv.split(self.y, self.X_actual, **self.split_params))
        if self.validation == "cv":
            self._run_shared_round_cross_validation(trial, params, splits)
            return
        all_test_scores: list[dict[str, float | str] | float | str] = []
        all_train_scores: list[dict[str, float | str] | float | str] = []
        all_fit_times: list[float] = []
        all_score_times: list[float] = []
        failed_splits: list[int] = []
        first_exception: Exception | None = None

        for split_idx, (train, test) in enumerate(splits):
            fold_forecaster = clone(cloned_forecaster)

            y_train, X_actual_train = _safe_split(fold_forecaster, self.y, self.X_actual, train)
            y_test, X_actual_test = _safe_split(fold_forecaster, self.y, self.X_actual, test, train)
            X_forecast_train, X_forecast_test = _split_X_forecast(
                self.X_forecast,
                self.y,
                train,
                test,
            )

            # Adjust fit_params for this split
            fit_params = _check_method_params(self.y, params=self.fit_params, indices=train)
            score_params_test = _check_method_params(self.y, params=self.score_params, indices=test)

            try:
                # Fit
                fit_start = time.time()
                if self.coverage_rates is not None:
                    fit_params["coverage_rates"] = self.coverage_rates
                fold_forecaster.fit(
                    y=y_train,
                    X_actual=X_actual_train,
                    forecasting_horizon=self.forecasting_horizon,
                    X_future=self.X_future,
                    X_forecast=X_forecast_train,
                    **fit_params,
                )
                fit_time = time.time() - fit_start
                all_fit_times.append(fit_time)

                # Score test. yohou's ``_score`` takes precomputed predictions,
                # so predict first, then score.
                score_start = time.time()
                y_pred = _predict(
                    fold_forecaster,
                    y_test,
                    X_actual_test,
                    self.scorers,
                    predict_func_params=self.predict_func_params,
                    coverage_rates=self.coverage_rates,
                    X_future=self.X_future,
                    X_forecast=X_forecast_test,
                )
                test_scores = _score(
                    fold_forecaster,
                    y_train,
                    y_test,
                    y_pred,
                    self.scorers,
                    score_params_test,
                    self.error_score,
                )
                score_time = time.time() - score_start
                all_score_times.append(score_time)
                all_test_scores.append(test_scores)

                # Score train if requested, through yohou's recipe: the stretch ends
                # before the rows the forecaster held back (a split-conformal model's
                # calibration rows), positions are relative to the training window,
                # score params are sliced to the scored rows, and a window too short to
                # leave room scores NaN with a warning instead of scoring other rows.
                if self.return_train_score:
                    window = _train_window_predictions(
                        fold_forecaster,
                        y_train,
                        X_actual_train,
                        n_rows=len(test),
                        scorer=self.scorers,
                        predict_func_params=self.predict_func_params,
                        coverage_rates=self.coverage_rates,
                        X_future=self.X_future,
                        X_forecast_train=X_forecast_train,
                    )
                    train_scores = _score_train_window(
                        fold_forecaster,
                        window,
                        self.scorers,
                        y=self.y,
                        score_params=self.score_params,
                        train=train,
                        error_score=self.error_score,
                    )
                    all_train_scores.append(train_scores)

            except Exception as exc:
                if self.error_score == "raise":
                    raise
                if first_exception is None:
                    first_exception = exc
                failed_splits.append(split_idx)
                error_val = float(self.error_score) if isinstance(self.error_score, numbers.Number) else np.nan
                multimetric = isinstance(self.scorers, _MultimetricScorer)
                # One entry per fold, even when the failure struck after part of the
                # fold was already recorded: a fold that fit and then failed scoring
                # would otherwise append a second fit time and a second test score,
                # skewing every mean computed over them.
                if len(all_test_scores) <= split_idx:
                    all_test_scores.append(
                        dict.fromkeys(self.scorers._scorers, error_val) if multimetric else error_val
                    )
                if self.return_train_score and len(all_train_scores) <= split_idx:
                    all_train_scores.append(
                        dict.fromkeys(self.scorers._scorers, error_val) if multimetric else error_val
                    )
                if len(all_fit_times) <= split_idx:
                    all_fit_times.append(0.0)
                if len(all_score_times) <= split_idx:
                    all_score_times.append(0.0)

        if failed_splits:
            assert first_exception is not None  # noqa: S101  # set with the first failed split
            # The same keys the trial-level handler uses, so a consumer reads one
            # shape whether a failure was absorbed in the fold loop or escaped it.
            # `failed_splits` is the explicit mark that the trial's score carries
            # absorbed failures; it is never inferred from the score's value, which
            # a numeric error_score would make finite.
            trial.set_user_attr("exception", str(first_exception))
            trial.set_user_attr("exception_type", type(first_exception).__name__)
            trial.set_user_attr("failed_splits", failed_splits)
            logger.warning(
                "Trial %d: %d of %d fold(s) failed (splits %s), first failure %s: %s",
                trial.number,
                len(failed_splits),
                len(splits),
                failed_splits,
                type(first_exception).__name__,
                first_exception,
            )

        # Store results as trial user attributes
        self._store_scores(trial, all_test_scores, all_train_scores)
        self._store_timing(trial, all_fit_times, all_score_times)

    def _run_shared_round_cross_validation(
        self,
        trial: optuna.trial.Trial,
        params: dict[str, Any],
        splits: list[tuple[np.ndarray, np.ndarray]],
    ) -> None:
        """Evaluate the trial with ``validation="cv"`` and store its results on the trial.

        yohou's ``_evaluate_candidate_shared_rounds`` fits every fold with its
        test window as the evaluation set, chooses one round per estimator from
        the fold-average stopping curve, and scores every fold cut to that
        round, exactly as ``GridSearchCV(validation="cv")`` evaluates one
        candidate. Its per-fold results map onto the same user attributes as the
        default loop, and its round record is stored beside them.

        Parameters
        ----------
        trial : optuna.trial.Trial
            Optuna trial for storing results.
        params : dict
            Parameter settings for the forecaster.
        splits : list of tuple of np.ndarray
            ``(train, test)`` row indices for every fold.

        """
        try:
            results, record = self._evaluate_shared_rounds(params, splits)
        except Exception as exc:
            if self.error_score == "raise":
                raise
            # With a numeric error_score, a failed fold is absorbed inside the call and
            # recorded in its result, so what escapes is a configuration yohou refuses
            # before fitting any fold (dir-rec, dart boosting, CatBoost without an
            # explicit learning_rate, a validation_size on the forecaster, ...).
            raise _SharedRoundConfigurationError(str(exc)) from exc
        # yohou types the record and results as dict[str, object]; these are their documented shapes.
        rounds = cast("dict[str, int]", record["rounds"])
        boundary_positions = cast("list[str]", record["boundary_positions"])
        curve_lengths = cast("list[dict[str, int] | None]", record["curve_lengths"])
        # Plain JSON types: a storage backend serialises user attributes.
        trial.set_user_attr("rounds", {str(k): int(v) for k, v in rounds.items()})
        trial.set_user_attr("rounds_at_boundary", bool(record["rounds_at_boundary"]))
        trial.set_user_attr("boundary_positions", [str(p) for p in boundary_positions])
        trial.set_user_attr(
            "curve_lengths",
            [None if lengths is None else {str(k): int(v) for k, v in lengths.items()} for lengths in curve_lengths],
        )

        failed_splits = [i for i, result in enumerate(results) if result.get("fit_error") is not None]
        if failed_splits:
            fit_error = cast("str", results[failed_splits[0]]["fit_error"])
            exception_type, exception = _exception_from_traceback(fit_error)
            # The same keys the default loop sets, so a consumer reads one shape.
            trial.set_user_attr("exception", exception)
            trial.set_user_attr("exception_type", exception_type)
            trial.set_user_attr("failed_splits", failed_splits)
            logger.warning(
                "Trial %d: %d of %d fold(s) failed (splits %s), first failure %s: %s",
                trial.number,
                len(failed_splits),
                len(splits),
                failed_splits,
                exception_type,
                exception,
            )

        self._store_scores(
            trial,
            [result["test_scores"] for result in results],
            [result["train_scores"] for result in results] if self.return_train_score else [],
        )
        self._store_timing(
            trial,
            [cast("float", result["fit_time"]) for result in results],
            [cast("float", result["score_time"]) for result in results],
        )

    def _evaluate_shared_rounds(
        self, params: dict[str, Any], splits: list[tuple[np.ndarray, np.ndarray]]
    ) -> tuple[list[dict[str, object]], dict[str, object]]:
        """Run yohou's shared-round evaluation of one candidate on every split.

        Parameters
        ----------
        params : dict
            Parameter settings for the forecaster.
        splits : list of tuple of np.ndarray
            ``(train, test)`` row indices for every fold.

        Returns
        -------
        results : list of dict
            One ``_fit_and_score``-shaped result per fold, in ``splits`` order.
        record : dict
            The round record: ``rounds``, ``rounds_at_boundary``,
            ``boundary_positions`` and ``curve_lengths``.

        """
        return _evaluate_candidate_shared_rounds(
            self.forecaster,
            self.y,
            self.X_actual,
            self.forecasting_horizon,
            X_future=self.X_future,
            X_forecast=self.X_forecast,
            splits=splits,
            parameters=params,
            early_stopping_adapter=None if self.early_stopping_adapter is None else clone(self.early_stopping_adapter),
            scorer=self.scorers,
            verbose=self.verbose,
            fit_params=self.fit_params,
            predict_func_params=self.predict_func_params,
            score_params=self.score_params,
            return_train_score=self.return_train_score,
            return_times=True,
            error_score=self.error_score,
            coverage_rates=self.coverage_rates,
        )

    def _store_scores(
        self,
        trial: optuna.trial.Trial,
        all_test_scores: list[Any],
        all_train_scores: list[Any],
    ) -> None:
        """Store cross-validation scores as trial user attributes.

        Parameters
        ----------
        trial : optuna.trial.Trial
            Optuna trial for storing results.
        all_test_scores : list
            Test scores for each fold.
        all_train_scores : list
            Train scores for each fold (may be empty).

        """
        if self.multimetric:
            self._store_multimetric_scores(trial, all_test_scores, all_train_scores)
        else:
            self._store_single_metric_scores(trial, all_test_scores, all_train_scores)

    def _store_multimetric_scores(
        self,
        trial: optuna.trial.Trial,
        all_test_scores: list[Any],
        all_train_scores: list[Any],
    ) -> None:
        """Store multi-metric scores as trial user attributes.

        Parameters
        ----------
        trial : optuna.trial.Trial
            Trial to store results on.
        all_test_scores : list
            Test scores for each fold (each entry is a dict).
        all_train_scores : list
            Train scores for each fold.

        """
        # Collect metric names from the first score dict
        if not all_test_scores or not isinstance(all_test_scores[0], dict):
            return

        metric_names = list(all_test_scores[0].keys())

        for metric_name in metric_names:
            test_vals = []
            for i, scores in enumerate(all_test_scores):
                val = scores[metric_name] if isinstance(scores, dict) else np.nan
                val = float(val) if isinstance(val, numbers.Number) else np.nan
                trial.set_user_attr(f"split{i}_test_{metric_name}", val)
                test_vals.append(val)

            # A plain mean, never nanmean: the folds are shared across trials, and the
            # folds a trial fails are usually the harder ones, so a mean over only the
            # surviving folds is optimistic. A failed fold contributes its error_score,
            # and the NaN default sinks the trial, matching sklearn.
            mean_val = float(np.mean(test_vals))
            trial.set_user_attr(f"mean_test_{metric_name}", mean_val)

            if self.return_train_score and all_train_scores:
                train_vals = []
                for i, scores in enumerate(all_train_scores):
                    val = scores[metric_name] if isinstance(scores, dict) else np.nan
                    val = float(val) if isinstance(val, numbers.Number) else np.nan
                    trial.set_user_attr(f"split{i}_train_{metric_name}", val)
                    train_vals.append(val)

                mean_train = float(np.mean(train_vals))
                trial.set_user_attr(f"mean_train_{metric_name}", mean_train)

    def _store_single_metric_scores(
        self,
        trial: optuna.trial.Trial,
        all_test_scores: list[Any],
        all_train_scores: list[Any],
    ) -> None:
        """Store single metric scores as trial user attributes.

        Parameters
        ----------
        trial : optuna.trial.Trial
            Trial to store results on.
        all_test_scores : list
            Test scores for each fold (each entry is a float).
        all_train_scores : list
            Train scores for each fold.

        """
        test_vals = []
        for i, score in enumerate(all_test_scores):
            val = float(score) if isinstance(score, numbers.Number) else np.nan
            trial.set_user_attr(f"split{i}_test_score", val)
            test_vals.append(val)

        # A plain mean, never nanmean: see _store_multimetric_scores.
        mean_test = float(np.mean(test_vals))
        trial.set_user_attr("mean_test_score", mean_test)

        if self.return_train_score and all_train_scores:
            train_vals = []
            for i, score in enumerate(all_train_scores):
                val = float(score) if isinstance(score, numbers.Number) else np.nan
                trial.set_user_attr(f"split{i}_train_score", val)
                train_vals.append(val)

            mean_train = float(np.mean(train_vals))
            trial.set_user_attr("mean_train_score", mean_train)

    def _store_timing(
        self,
        trial: optuna.trial.Trial,
        fit_times: list[float],
        score_times: list[float],
    ) -> None:
        """Store timing information as trial user attributes.

        Parameters
        ----------
        trial : optuna.trial.Trial
            Trial to store timing info on.
        fit_times : list of float
            Fit time for each fold.
        score_times : list of float
            Score time for each fold.

        """
        trial.set_user_attr("mean_fit_time", float(np.mean(fit_times)))
        trial.set_user_attr("std_fit_time", float(np.std(fit_times)))
        trial.set_user_attr("mean_score_time", float(np.mean(score_times)))
        trial.set_user_attr("std_score_time", float(np.std(score_times)))

    def _get_primary_metric(self, trial: optuna.trial.Trial) -> float:
        """Get the primary metric value for optimization.

        Parameters
        ----------
        trial : optuna.trial.Trial
            Trial to read metric from.

        Returns
        -------
        float
            Primary metric value, or -inf if NaN.

        """
        if self.refit and isinstance(self.refit, str):
            metric_to_optimize = trial.user_attrs.get(f"mean_test_{self.refit}", float("nan"))
        else:
            # Use first scorer from available metrics
            test_keys = [k for k in trial.user_attrs if k.startswith("mean_test_")]
            if not test_keys:
                return float("-inf")
            metric_to_optimize = trial.user_attrs[test_keys[0]]

        if np.isnan(metric_to_optimize):
            return float("-inf")
        return metric_to_optimize

    def _handle_error(self, trial: optuna.trial.Trial, exception: Exception) -> float:
        """Handle exceptions during trial evaluation.

        Parameters
        ----------
        trial : optuna.trial.Trial
            Trial that encountered an error.
        exception : Exception
            The exception that was raised.

        Returns
        -------
        float
            Error score value, or -inf if NaN.

        Raises
        ------
        Exception
            If error_score is 'raise'.

        """
        trial.set_user_attr("exception", str(exception))
        trial.set_user_attr("exception_type", type(exception).__name__)

        if isinstance(self.error_score, str) and self.error_score == "raise":
            raise exception

        error_value = self.error_score
        assert isinstance(error_value, int | float)  # noqa: S101  # internal type-narrowing invariant

        if np.isnan(error_value):
            return float("-inf")

        return float(error_value)


def _exception_from_traceback(traceback: str) -> tuple[str, str]:
    """The exception type name and message on the last line of a formatted traceback.

    yohou records a failed fold's fit as ``traceback.format_exc()`` text rather
    than the exception object; this recovers what the default loop stores.

    Parameters
    ----------
    traceback : str
        A formatted traceback.

    Returns
    -------
    tuple of (str, str)
        The exception's class name (without its module path) and its message.

    Examples
    --------
    >>> _exception_from_traceback("sklearn.exceptions.NotFittedError: not fitted")
    ('NotFittedError', 'not fitted')

    """
    lines = [line for line in traceback.strip().splitlines() if line.strip()]
    last = lines[-1] if lines else ""
    name, _, message = last.partition(": ")
    return name.rsplit(".", 1)[-1], message
