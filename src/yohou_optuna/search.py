"""Optuna-powered hyperparameter search for Yohou forecasters."""

from __future__ import annotations

import time
from numbers import Integral, Real

import numpy as np
import optuna
from optuna.distributions import BaseDistribution
from sklearn.base import clone
from sklearn.utils._param_validation import Interval
from sklearn.utils.metadata_routing import (
    _raise_for_params,
)
from sklearn.utils.validation import _check_method_params, indexable
from sklearn_optuna.optuna import Callback, Sampler, Storage
from yohou.model_selection.search import BaseSearchCV
from yohou.model_selection.split import check_cv
from yohou.utils import validate_search_data

from .objective import _Objective
from .utils import _build_cv_results


class OptunaSearchCV(BaseSearchCV):
    """Hyperparameter search using Optuna optimization for Yohou forecasters.

    OptunaSearchCV uses Optuna's trial-based optimization framework
    to search for the best hyperparameters of a yohou forecaster.
    It overrides ``fit()`` to manage the Optuna study lifecycle and
    uses yohou's time series cross-validation for evaluation.

    Parameters
    ----------
    forecaster : BaseForecaster
        A yohou forecaster instance to optimize.
    param_distributions : dict[str, optuna.distributions.BaseDistribution]
        Dictionary with parameter names as keys and Optuna distribution
        objects as values.  Distributions define the search space for each
        hyperparameter.
    scoring : BaseScorer or dict of str to BaseScorer
        Scoring function(s) for evaluation.  Must be a yohou BaseScorer
        instance or a dictionary mapping names to BaseScorer instances.
    sampler : Sampler or None, default=None
        A wrapped Optuna sampler.  If ``None``, TPESampler is used.
    storage : Storage or None, default=None
        A wrapped Optuna storage.  If ``None``, in-memory storage is used.
    callbacks : dict of str to Callback or None, default=None
        Dictionary mapping callback names to Callback instances.
    n_trials : int, default=10
        Number of trials for hyperparameter search.
    timeout : float or None, default=None
        Stop study after the given number of seconds.
    n_jobs : int or None, default=None
        Number of parallel trials to run via Optuna's threading.
        ``None`` or ``1`` runs sequentially.  ``-1`` uses all cores.
    refit : bool, str, or callable, default=True
        Refit a forecaster using the best found parameters on the
        whole dataset.
    cv : int, splitter, or None, default=None
        Cross-validation splitting strategy.
    verbose : int, default=0
        Controls the verbosity.
    pre_dispatch : int or str, default='2*n_jobs'
        Controls the number of jobs dispatched during parallel execution.
        Not directly used by Optuna but kept for API compatibility.
    error_score : 'raise' or numeric, default=np.nan
        Value to assign to the score if an error occurs.
    return_train_score : bool, default=False
        Whether to include training scores in ``cv_results_``.

    Attributes
    ----------
    cv_results_ : dict of numpy ndarrays
        Cross-validation results dictionary.
    best_forecaster_ : BaseForecaster
        Forecaster refitted on the full dataset with best parameters.
    best_score_ : float
        Mean cross-validated score of the best forecaster.
    best_params_ : dict
        Best parameter setting found.
    best_index_ : int
        Index of the best parameter combination in ``cv_results_``.
    scorer_ : BaseScorer or dict
        Scorer function(s) used.
    n_splits_ : int
        Number of cross-validation splits.
    multimetric_ : bool
        Whether multiple metrics were used.
    study_ : optuna.study.Study
        The Optuna study containing all trial information.
    trials_ : list of optuna.trial.FrozenTrial
        All trials executed during the search.

    See Also
    --------
    yohou.model_selection.GridSearchCV : Exhaustive grid search.
    yohou.model_selection.RandomizedSearchCV : Randomized search.

    Examples
    --------
    >>> import optuna
    >>> optuna.logging.set_verbosity(optuna.logging.WARNING)
    >>> from datetime import datetime, timedelta
    >>> import polars as pl
    >>> from optuna.distributions import FloatDistribution
    >>> from sklearn.linear_model import Ridge
    >>> from yohou.metrics import MeanAbsoluteError
    >>> from yohou.point import PointReductionForecaster
    >>> from yohou_optuna import OptunaSearchCV, Sampler
    >>> dates = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(100)]
    >>> y = pl.DataFrame({"time": dates, "value": range(100)})
    >>> param_distributions = {
    ...     "estimator__alpha": FloatDistribution(0.01, 10.0, log=True),
    ... }
    >>> search = OptunaSearchCV(
    ...     PointReductionForecaster(estimator=Ridge()),
    ...     param_distributions,
    ...     scoring=MeanAbsoluteError(),
    ...     n_trials=2,
    ...     cv=2,
    ...     sampler=Sampler(sampler=optuna.samplers.TPESampler, seed=42),
    ... )
    >>> search.fit(y, forecasting_horizon=3)  # doctest: +ELLIPSIS
    OptunaSearchCV(...)

    """

    _parameter_constraints: dict = {
        **BaseSearchCV._parameter_constraints,
        "param_distributions": [dict],
        "n_trials": [Interval(Integral, 1, None, closed="left"), None],
        "timeout": [Interval(Real, 0, None, closed="neither"), None],
        "sampler": [Sampler, None],
        "storage": [Storage, None],
        "callbacks": [dict, None],
    }

    def __init__(
        self,
        forecaster,
        param_distributions,
        *,
        scoring=None,
        sampler=None,
        storage=None,
        callbacks=None,
        n_trials=10,
        timeout=None,
        n_jobs=None,
        refit=True,
        cv=None,
        verbose=0,
        pre_dispatch="2*n_jobs",
        error_score=np.nan,
        return_train_score=False,
    ):
        super().__init__(
            forecaster=forecaster,
            scoring=scoring,
            n_jobs=n_jobs,
            refit=refit,
            cv=cv,
            verbose=verbose,
            pre_dispatch=pre_dispatch,
            error_score=error_score,
            return_train_score=return_train_score,
        )
        self.param_distributions = param_distributions
        self.n_trials = n_trials
        self.sampler = sampler
        self.storage = storage
        self.timeout = timeout
        self.callbacks = callbacks

    def _run_search(self, evaluate_candidates):
        """Not used. OptunaSearchCV overrides fit() directly.

        Parameters
        ----------
        evaluate_candidates : callable
            Unused callback.

        Raises
        ------
        NotImplementedError
            Always, since OptunaSearchCV overrides fit().

        """
        raise NotImplementedError("OptunaSearchCV overrides fit() directly.")

    def fit(self, y, X=None, forecasting_horizon=1, *, study=None, **params) -> OptunaSearchCV:
        """Run Optuna hyperparameter optimization.

        Parameters
        ----------
        y : pl.DataFrame
            Target time series with a ``"time"`` column.
        X : pl.DataFrame or None, default=None
            Exogenous features with a ``"time"`` column.
        forecasting_horizon : int, default=1
            Number of steps ahead to forecast.
        study : optuna.study.Study or None, default=None
            An existing Optuna study to continue from.  If ``None``, a new
            study is created.
        **params : dict
            Parameters passed to fit, predict, and score methods
            via metadata routing.

        Returns
        -------
        self
            Fitted search instance.

        Raises
        ------
        ValueError
            If ``param_distributions`` contains non-BaseDistribution values.
        TypeError
            If ``callbacks`` is not a dict or contains non-Callback values.

        """
        _raise_for_params(params, self, "fit")

        # Validate input data
        validate_search_data(y, X)

        scorers, refit_metric = self._get_scorers()

        y, X = indexable(y, X)
        params = _check_method_params(y, params=params)

        self.scorer_ = scorers
        self.multimetric_ = isinstance(self.scoring, dict)

        # Validate param_distributions
        for param_name, distribution in self.param_distributions.items():
            if not isinstance(distribution, BaseDistribution):
                msg = (
                    f"Parameter '{param_name}' has an invalid distribution. "
                    f"Expected optuna.distributions.BaseDistribution, got {type(distribution)}."
                )
                raise ValueError(msg)

        # Get routed params
        routed_params = self._get_routed_params_for_fit(params)

        # Get CV splitter
        cv_orig = check_cv(self.cv, forecasting_horizon)
        self.n_splits_ = cv_orig.get_n_splits(y, X, **routed_params.splitter.split)

        # Instantiate sampler from wrapper
        sampler_instance = None
        if self.sampler is not None:
            sampler_instance = self.sampler.instantiate().instance_

        # Instantiate storage from wrapper
        storage_instance = None
        if self.storage is not None:
            storage_instance = self.storage.instantiate().instance_

        # Prepare callbacks
        callback_list = None
        if self.callbacks is not None:
            if not isinstance(self.callbacks, dict):
                msg = f"callbacks must be a dict of str to Callback, got {type(self.callbacks)}"
                raise TypeError(msg)
            callback_list = []
            for name, callback in self.callbacks.items():
                if not isinstance(callback, Callback):
                    msg = f"Callback '{name}' must be a Callback instance, got {type(callback)}"
                    raise TypeError(msg)
                callback.instantiate()
                callback_list.append(callback)

        # Create or reuse study
        if study is not None:
            optuna_study = study
            if sampler_instance is not None:
                optuna_study.sampler = sampler_instance
        else:
            optuna_study = optuna.create_study(
                direction="maximize",
                sampler=sampler_instance,
                storage=storage_instance,
            )

        # Create objective
        objective = _Objective(
            forecaster=self.forecaster,
            param_distributions=self.param_distributions,
            y=y,
            X=X,
            forecasting_horizon=forecasting_horizon,
            cv=cv_orig,
            scorers=scorers,
            fit_params=routed_params.forecaster.fit,
            predict_params=routed_params.forecaster.predict,
            score_params=routed_params.scorer.score,
            verbose=self.verbose,
            return_train_score=self.return_train_score,
            error_score=self.error_score,
            multimetric=self.multimetric_,
            refit=self.refit,
        )

        # Run optimization
        optuna_study.optimize(
            objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            callbacks=callback_list,
            n_jobs=self.n_jobs if self.n_jobs is not None else 1,
        )

        # Store study and trials
        self.study_ = optuna_study
        self.trials_ = optuna_study.trials

        # Build cv_results_ from trials
        self.cv_results_ = _build_cv_results(self.trials_, self.multimetric_, self.return_train_score)

        # Handle empty results
        if not self.cv_results_["params"] or len(self.cv_results_["params"]) == 0:
            if self.refit:
                msg = "No trials were completed. 'refit' cannot be true."
                raise ValueError(msg)
            return self

        # Select best index and params
        if self.refit or not self.multimetric_:
            self.best_index_ = self._select_best_index(self.refit, refit_metric, self.cv_results_)
            if not callable(self.refit):
                self.best_score_ = self.cv_results_[f"mean_test_{refit_metric}"][self.best_index_]
            self.best_params_ = self.cv_results_["params"][self.best_index_]

        # Refit best forecaster on full data
        if self.refit:
            self.best_forecaster_ = clone(self.forecaster).set_params(**clone(self.best_params_, safe=False))
            refit_start_time = time.time()
            fit_params_filtered = {k: v for k, v in routed_params.forecaster.fit.items() if k not in ["y", "X"]}
            self.best_forecaster_.fit(y, X, forecasting_horizon, **fit_params_filtered)
            self.refit_time_ = time.time() - refit_start_time

        return self

    def __sklearn_tags__(self):
        """Get tags for this search estimator.

        Adds ``search_type = "optuna"`` to the tags returned by
        ``BaseSearchCV.__sklearn_tags__``.  This allows downstream code
        to distinguish Optuna-based searches from grid/random searches.

        Returns
        -------
        Tags
            Estimator tags with ``search_type`` set to ``"optuna"``.

        """
        tags = super().__sklearn_tags__()
        assert tags.forecaster_tags is not None
        tags.forecaster_tags.search_type = "optuna"
        return tags
