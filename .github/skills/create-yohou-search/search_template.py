"""Module docstring.

Custom hyperparameter search extending yohou's BaseSearchCV.
"""

import numbers

import numpy as np
from sklearn.utils._param_validation import Interval
from yohou.model_selection.search import BaseSearchCV


class MyCustomSearch(BaseSearchCV):
    """Custom hyperparameter search for time series forecasters.

    Extends yohou's BaseSearchCV which inherits from BaseForecaster.
    The only abstract method to implement is ``_run_search(evaluate_candidates)``.

    Parameters
    ----------
    forecaster : BaseForecaster
        Yohou forecaster to tune.
    param_distributions : dict
        Parameter space (custom format for your sampler).
    n_iter : int, default=10
        Number of parameter settings to sample.
    scoring : BaseScorer, default=None
        Yohou scorer for evaluation.
    cv : BaseSplitter, default=None
        Time series cross-validator.
    n_jobs : int, default=None
        Number of parallel jobs.
    refit : bool, default=True
        Whether to refit on full data with best params.
    verbose : int, default=0
        Verbosity level.
    pre_dispatch : str, default="2*n_jobs"
        Controls parallel dispatch.
    error_score : float, default=np.nan
        Value to assign on error.
    return_train_score : bool, default=False
        Whether to compute training scores.
    random_state : int, default=None
        Random seed for reproducibility.

    Examples
    --------
    >>> from yohou.point import PointReductionForecaster
    >>> from yohou.metrics import MeanAbsoluteError
    >>> search = MyCustomSearch(
    ...     forecaster=PointReductionForecaster(),
    ...     param_distributions={"estimator__alpha": [0.1, 1.0, 10.0]},
    ...     scoring=MeanAbsoluteError(),
    ... )
    """

    _parameter_constraints: dict = {
        **BaseSearchCV._parameter_constraints,
        "param_distributions": [dict],
        "n_iter": [Interval(numbers.Integral, 1, None, closed="left")],
        "random_state": [numbers.Integral, None],
    }

    def __init__(
        self,
        forecaster,
        param_distributions,
        *,
        n_iter=10,
        scoring=None,
        n_jobs=None,
        refit=True,
        cv=None,
        verbose=0,
        pre_dispatch="2*n_jobs",
        error_score=np.nan,
        return_train_score=False,
        random_state=None,
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
        self.n_iter = n_iter
        self.random_state = random_state

    def _run_search(self, evaluate_candidates):
        """Run the custom search strategy.

        This is the only abstract method you MUST implement. The
        ``evaluate_candidates`` callback handles all CV scoring internally.

        Parameters
        ----------
        evaluate_candidates : callable
            A function that takes a list of candidate parameter dicts and
            evaluates them via cross-validation. Call it as:
            ``evaluate_candidates(candidate_params)``
            where candidate_params is a list of dicts.

        """
        # Example: Random sampling from param_distributions
        rng = np.random.RandomState(self.random_state)
        candidates = []
        for _ in range(self.n_iter):
            candidate = {}
            for param_name, param_values in self.param_distributions.items():
                candidate[param_name] = rng.choice(param_values)
            candidates.append(candidate)

        # evaluate_candidates handles CV, scoring, and results tracking
        evaluate_candidates(candidates)
