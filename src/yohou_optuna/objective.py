"""Objective function for Optuna hyperparameter optimization in Yohou."""

from __future__ import annotations

import logging
import numbers
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import optuna
import polars as pl
from sklearn.base import clone
from yohou.base import BaseForecaster
from yohou.metrics.base import BaseScorer
from yohou.model_selection.utils import (
    _evaluate_candidate_shared_rounds,
    _fit_fold,
    _MultimetricScorer,
    _score_fold,
    _score_train_window,
    _train_window_predictions,
)

logger = logging.getLogger(__name__)


@dataclass
class _FoldOutcome:
    """What one fold contributes to a trial.

    Attributes
    ----------
    test_scores : dict, float or str
        The fold's test score, or ``error_score`` when it failed.
    train_scores : dict, float, str or None
        The fold's train score when train scores are requested, else ``None``.
    fit_time : float
        Seconds spent fitting, up to the error for a failed fit.
    score_time : float
        Seconds spent predicting and scoring the test window, ``0.0`` when it failed.
    failure : tuple of (str, str) or None
        The first failure's exception type name and message, or ``None``.

    """

    test_scores: Any
    train_scores: Any
    fit_time: float
    score_time: float
    failure: tuple[str, str] | None = None


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

        Every fold is fitted and scored by yohou's own fold evaluation, so a
        trial scores its parameters as yohou's ``GridSearchCV`` scores the same
        candidate: ``_fit_fold`` and ``_score_fold`` by default, and
        ``_evaluate_candidate_shared_rounds`` with ``validation="cv"``. Each fold
        yields one outcome, and the outcomes are recorded on the trial the same
        way in both modes.

        Parameters
        ----------
        trial : optuna.trial.Trial
            Optuna trial for storing results.
        params : dict
            Parameter settings for the forecaster.

        """
        splits = list(self.cv.split(self.y, self.X_actual, **self.split_params))
        if self.validation == "cv":
            outcomes = self._shared_round_outcomes(trial, params, splits)
        else:
            outcomes = [
                self._evaluate_fold(params, train, test, split_idx, len(splits))
                for split_idx, (train, test) in enumerate(splits)
            ]
        self._record_folds(trial, outcomes)

    def _evaluate_fold(
        self, params: dict[str, Any], train: np.ndarray, test: np.ndarray, split_idx: int, n_splits: int
    ) -> _FoldOutcome:
        """Fit and score one fold through yohou's fold evaluation.

        The test side is scored by ``_score_fold``, and the train side by the
        same helpers ``_score_fold`` uses, in a separate step: a failure while
        train scoring then keeps the fold's real test score and fails only the
        train side, where scoring both in one call would lose it.

        Parameters
        ----------
        params : dict
            Parameter settings for the forecaster.
        train, test : np.ndarray
            The fold's training and test row indices.
        split_idx : int
            Zero-based index of the fold.
        n_splits : int
            Number of folds, for verbose progress.

        Returns
        -------
        _FoldOutcome
            The fold's scores, times, and first failure, if any.

        Raises
        ------
        Exception
            Any failure, when ``error_score`` is ``"raise"``.

        """
        fold = _fit_fold(
            clone(self.forecaster),
            self.y,
            self.X_actual,
            self.forecasting_horizon,
            X_future=self.X_future,
            X_forecast=self.X_forecast,
            scorer=self.scorers,
            train=train,
            test=test,
            verbose=self.verbose,
            parameters=params,
            fit_params=self.fit_params,
            score_params=self.score_params,
            return_train_score=self.return_train_score,
            split_progress=(split_idx, n_splits),
            candidate_progress=None,
            error_score=self.error_score,
            coverage_rates=self.coverage_rates,
        )
        if fold.fit_error is not None:
            # yohou filled the fold's scores with error_score when it recorded the failure.
            return _FoldOutcome(
                test_scores=fold.test_scores,
                train_scores=fold.train_scores,
                fit_time=fold.fit_time,
                score_time=0.0,
                failure=_exception_from_traceback(fold.fit_error),
            )

        try:
            result = _score_fold(
                fold,
                X_future=self.X_future,
                scorer=self.scorers,
                verbose=self.verbose,
                predict_func_params=self.predict_func_params,
                return_train_score=False,
                return_parameters=False,
                return_n_test_samples=False,
                return_times=True,
                return_forecaster=False,
                return_predictions=False,
                predict_forecasting_horizon=None,
                predict_stride=None,
                predict_method=None,
                error_score=self.error_score,
                coverage_rates=self.coverage_rates,
            )
        except Exception as exc:
            if self.error_score == "raise":
                raise
            error = self._error_scores()
            return _FoldOutcome(
                test_scores=error,
                train_scores=error if self.return_train_score else None,
                fit_time=fold.fit_time,
                score_time=0.0,
                failure=(type(exc).__name__, str(exc)),
            )

        outcome = _FoldOutcome(
            test_scores=result["test_scores"],
            train_scores=None,
            fit_time=cast("float", result["fit_time"]),
            score_time=cast("float", result["score_time"]),
        )
        if self.return_train_score:
            try:
                window = _train_window_predictions(
                    fold.forecaster,
                    fold.y_train,
                    fold.X_actual_train,
                    n_rows=len(test),
                    scorer=self.scorers,
                    predict_func_params=self.predict_func_params,
                    coverage_rates=self.coverage_rates,
                    X_future=self.X_future,
                    X_forecast_train=fold.X_forecast_train,
                )
                outcome.train_scores = _score_train_window(
                    fold.forecaster,
                    window,
                    self.scorers,
                    y=self.y,
                    score_params=self.score_params,
                    train=train,
                    error_score=self.error_score,
                )
            except Exception as exc:
                if self.error_score == "raise":
                    raise
                outcome.train_scores = self._error_scores()
                outcome.failure = (type(exc).__name__, str(exc))
        return outcome

    def _shared_round_outcomes(
        self,
        trial: optuna.trial.Trial,
        params: dict[str, Any],
        splits: list[tuple[np.ndarray, np.ndarray]],
    ) -> list[_FoldOutcome]:
        """Evaluate the trial with ``validation="cv"``, store its round record, and return its fold outcomes.

        yohou's ``_evaluate_candidate_shared_rounds`` fits every fold with its
        test window as the evaluation set, chooses one round per estimator from
        the fold-average stopping curve, and scores every fold cut to that
        round, exactly as ``GridSearchCV(validation="cv")`` evaluates one
        candidate.

        Parameters
        ----------
        trial : optuna.trial.Trial
            Optuna trial for storing the round record.
        params : dict
            Parameter settings for the forecaster.
        splits : list of tuple of np.ndarray
            ``(train, test)`` row indices for every fold.

        Returns
        -------
        list of _FoldOutcome
            One outcome per fold, in ``splits`` order.

        Raises
        ------
        _SharedRoundConfigurationError
            If yohou refuses the trial's configuration for ``validation="cv"``.

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
        outcomes = []
        for result in results:
            fit_error = result.get("fit_error")
            outcomes.append(
                _FoldOutcome(
                    test_scores=result["test_scores"],
                    train_scores=result.get("train_scores"),
                    fit_time=cast("float", result["fit_time"]),
                    score_time=cast("float", result["score_time"]),
                    failure=None if fit_error is None else _exception_from_traceback(cast("str", fit_error)),
                )
            )
        return outcomes

    def _record_folds(self, trial: optuna.trial.Trial, outcomes: list[_FoldOutcome]) -> None:
        """Store the fold outcomes on the trial: scores, timing, and any failed splits.

        Parameters
        ----------
        trial : optuna.trial.Trial
            Optuna trial for storing results.
        outcomes : list of _FoldOutcome
            One outcome per fold.

        """
        failed_splits = [i for i, outcome in enumerate(outcomes) if outcome.failure is not None]
        if failed_splits:
            exception_type, exception = cast("tuple[str, str]", outcomes[failed_splits[0]].failure)
            # The same keys the trial-level handler uses, so a consumer reads one
            # shape whether a failure was absorbed in a fold or escaped the trial.
            # `failed_splits` is the explicit mark that the trial's score carries
            # absorbed failures; it is never inferred from the score's value, which
            # a numeric error_score would make finite.
            trial.set_user_attr("exception", exception)
            trial.set_user_attr("exception_type", exception_type)
            trial.set_user_attr("failed_splits", failed_splits)
            logger.warning(
                "Trial %d: %d of %d fold(s) failed (splits %s), first failure %s: %s",
                trial.number,
                len(failed_splits),
                len(outcomes),
                failed_splits,
                exception_type,
                exception,
            )

        self._store_scores(
            trial,
            [outcome.test_scores for outcome in outcomes],
            [outcome.train_scores for outcome in outcomes] if self.return_train_score else [],
        )
        self._store_timing(trial, [o.fit_time for o in outcomes], [o.score_time for o in outcomes])

    def _error_scores(self) -> dict[str, float] | float:
        """The scores a failed fold contributes, shaped like the scorer's output.

        Only called for an absorbed failure, so ``error_score`` is numeric here:
        ``"raise"`` propagates the failure before this is reached.

        Returns
        -------
        dict or float
            ``error_score`` per scorer for a multi-metric scorer, else ``error_score``.

        """
        error_value = float(cast("float", self.error_score))
        if isinstance(self.scorers, _MultimetricScorer):
            return dict.fromkeys(self.scorers._scorers, error_value)
        return error_value

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
