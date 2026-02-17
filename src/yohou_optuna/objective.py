"""Objective function for Optuna hyperparameter optimization in Yohou."""

from __future__ import annotations

import numbers
import time
import warnings
from typing import Any

import numpy as np
import optuna
import polars as pl
from sklearn.base import clone
from sklearn.utils.metaestimators import _safe_split
from sklearn.utils.validation import _check_method_params
from yohou.base import BaseForecaster
from yohou.metrics.base import BaseScorer
from yohou.model_selection.utils import _MultimetricScorer, _score


class _Objective:
    """Objective function for Optuna trials in OptunaSearchCV.

    This class encapsulates the logic for evaluating hyperparameter
    configurations during Optuna optimization. It handles parameter
    suggestion, cross-validation using yohou's time series scoring,
    and error handling.

    Parameters
    ----------
    forecaster : BaseForecaster
        Base forecaster to evaluate.
    param_distributions : dict[str, optuna.distributions.BaseDistribution]
        Dictionary mapping parameter names to Optuna distributions.
    y : pl.DataFrame
        Target time series.
    X : pl.DataFrame or None
        Feature time series.
    forecasting_horizon : int
        Number of steps ahead to forecast.
    cv : BaseSplitter
        Cross-validation splitter.
    scorers : BaseScorer or _MultimetricScorer
        Scoring functions.
    fit_params : dict
        Additional parameters passed to forecaster.fit().
    predict_params : dict
        Additional parameters passed to forecaster.predict().
    score_params : dict
        Additional parameters passed to scorer.
    verbose : int, default=0
        Verbosity level.
    return_train_score : bool, default=False
        Whether to include training scores.
    error_score : numeric or 'raise', default=np.nan
        Value to assign on error, or 'raise' to propagate exceptions.
    multimetric : bool, default=False
        Whether multiple metrics are being optimized.
    refit : bool or str, default=True
        Primary metric name for multi-metric optimization.

    """

    def __init__(
        self,
        forecaster: BaseForecaster,
        param_distributions: dict[str, Any],
        y: pl.DataFrame,
        X: pl.DataFrame | None,
        forecasting_horizon: int,
        cv: Any,
        scorers: BaseScorer | _MultimetricScorer,
        fit_params: dict[str, Any],
        predict_params: dict[str, Any],
        score_params: dict[str, Any],
        *,
        verbose: int = 0,
        return_train_score: bool = False,
        error_score: float | str = np.nan,
        multimetric: bool = False,
        refit: bool | str = True,
    ) -> None:
        self.forecaster = forecaster
        self.param_distributions = param_distributions
        self.y = y
        self.X = X
        self.forecasting_horizon = forecasting_horizon
        self.cv = cv
        self.scorers = scorers
        self.fit_params = fit_params
        self.predict_params = predict_params
        self.score_params = score_params
        self.verbose = verbose
        self.return_train_score = return_train_score
        self.error_score = error_score
        self.multimetric = multimetric
        self.refit = refit

    def __call__(self, trial: optuna.trial.Trial) -> float:
        """Evaluate a single trial.

        Parameters
        ----------
        trial : optuna.trial.Trial
            Optuna trial object for suggesting parameters.

        Returns
        -------
        float
            Optimization objective value (score to maximize).

        """
        # Suggest parameters
        study_params = self._suggest_parameters(trial)

        # Store parameters as user attributes
        self._store_parameters(trial, study_params)

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

        splits = list(self.cv.split(self.y, self.X))
        all_test_scores: list[dict[str, float | str] | float | str] = []
        all_train_scores: list[dict[str, float | str] | float | str] = []
        all_fit_times: list[float] = []
        all_score_times: list[float] = []

        for _split_idx, (train, test) in enumerate(splits):
            fold_forecaster = clone(cloned_forecaster)

            y_train, X_train = _safe_split(fold_forecaster, self.y, self.X, train)
            y_test, X_test = _safe_split(fold_forecaster, self.y, self.X, test, train)

            # Adjust fit_params for this split
            fit_params = _check_method_params(self.y, params=self.fit_params, indices=train)
            score_params_test = _check_method_params(self.y, params=self.score_params, indices=test)

            try:
                # Fit
                fit_start = time.time()
                fold_forecaster.fit(
                    y=y_train,
                    X=X_train,
                    forecasting_horizon=self.forecasting_horizon,
                    **fit_params,
                )
                fit_time = time.time() - fit_start
                all_fit_times.append(fit_time)

                # Score test
                score_start = time.time()
                test_scores = _score(
                    fold_forecaster,
                    y_train,
                    y_test,
                    X_test,
                    self.predict_params,
                    self.scorers,
                    score_params_test,
                    self.error_score,
                )
                score_time = time.time() - score_start
                all_score_times.append(score_time)
                all_test_scores.append(test_scores)

                # Score train if requested
                if self.return_train_score:
                    score_params_train = _check_method_params(self.y, params=self.score_params, indices=train)
                    train_reset = train[: -len(test)]
                    test_reset = train[-len(test) :]
                    y_train_reset, X_train_reset = _safe_split(fold_forecaster, y_train, X_train, train_reset)
                    y_train_test, X_train_test = _safe_split(fold_forecaster, y_train, X_train, test_reset, train_reset)
                    fold_forecaster.reset(y_train_reset, X_train_reset)
                    train_scores = _score(
                        fold_forecaster,
                        y_train_reset,
                        y_train_test,
                        X_train_test,
                        self.predict_params,
                        self.scorers,
                        score_params_train,
                        self.error_score,
                    )
                    all_train_scores.append(train_scores)

            except Exception:
                if self.error_score == "raise":
                    raise
                error_val = float(self.error_score) if isinstance(self.error_score, numbers.Number) else np.nan
                if isinstance(self.scorers, _MultimetricScorer):
                    error_scores: dict[str, float | str] | float = dict.fromkeys(self.scorers._scorers, error_val)
                    all_test_scores.append(error_scores)
                    if self.return_train_score:
                        all_train_scores.append(dict.fromkeys(self.scorers._scorers, error_val))
                else:
                    all_test_scores.append(error_val)
                    if self.return_train_score:
                        all_train_scores.append(error_val)
                all_fit_times.append(0.0)
                all_score_times.append(0.0)

        # Store results as trial user attributes
        self._store_scores(trial, all_test_scores, all_train_scores)
        self._store_timing(trial, all_fit_times, all_score_times)

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

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                mean_val = float(np.nanmean(test_vals))
            trial.set_user_attr(f"mean_test_{metric_name}", mean_val)

            if self.return_train_score and all_train_scores:
                train_vals = []
                for i, scores in enumerate(all_train_scores):
                    val = scores[metric_name] if isinstance(scores, dict) else np.nan
                    val = float(val) if isinstance(val, numbers.Number) else np.nan
                    trial.set_user_attr(f"split{i}_train_{metric_name}", val)
                    train_vals.append(val)

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    mean_train = float(np.nanmean(train_vals))
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

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mean_test = float(np.nanmean(test_vals))
        trial.set_user_attr("mean_test_score", mean_test)

        if self.return_train_score and all_train_scores:
            train_vals = []
            for i, score in enumerate(all_train_scores):
                val = float(score) if isinstance(score, numbers.Number) else np.nan
                trial.set_user_attr(f"split{i}_train_score", val)
                train_vals.append(val)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                mean_train = float(np.nanmean(train_vals))
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
        assert isinstance(error_value, int | float)

        if np.isnan(error_value):
            return float("-inf")

        return float(error_value)
