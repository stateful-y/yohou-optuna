"""Stub boosting estimator and adapter for testing ``validation="cv"`` without a boosting library.

Mirrors yohou's own test stubs. ``CurveRegressor`` behaves like a boosting
model whose prediction after r rounds is ``train_mean * r / 10``. Its
evaluation loss at round r is the mean absolute gap between the evaluation
targets and that prediction, so the best round depends on the evaluation data
it receives: a wrong evaluation set yields a different curve rather than
passing silently.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from yohou.model_selection import BaseEarlyStoppingAdapter


class CurveRegressor(RegressorMixin, BaseEstimator):
    """Boosting-like regressor with a controllable stopping curve.

    Parameters
    ----------
    n_rounds : int, default=60
        Round ceiling.
    patience : int, default=5
        Rounds without improvement before stopping on an evaluation set.
    fail_below_train_rows : int, default=0
        Raise when fitted on fewer training rows than this.

    """

    def __init__(self, n_rounds: int = 60, patience: int = 5, fail_below_train_rows: int = 0):
        self.n_rounds = n_rounds
        self.patience = patience
        self.fail_below_train_rows = fail_below_train_rows

    def fit(self, X, y, eval_set=None, **kwargs):
        """Fit, recording the stopping curve on ``eval_set`` when given.

        Parameters
        ----------
        X : array-like
            Training features.
        y : array-like
            Training targets.
        eval_set : list of tuple or None, default=None
            ``[(X_eval, y_eval)]``, as boosting libraries accept it.
        **kwargs : dict
            Ignored.

        Returns
        -------
        self
            The fitted estimator.

        """
        arr = np.asarray(y, dtype=float)
        if len(arr) < self.fail_below_train_rows:
            raise RuntimeError(f"CurveRegressor refuses {len(arr)} training rows")
        self._ncols = 1 if arr.ndim == 1 else arr.shape[1]
        self.train_mean_ = float(np.nanmean(arr))
        self.curve_ = []
        if eval_set is None:
            self.rounds_trained_ = self.n_rounds
        else:
            y_eval = np.asarray(eval_set[0][1], dtype=float).ravel()
            best, best_round = np.inf, 0
            for r in range(1, self.n_rounds + 1):
                loss = float(np.mean(np.abs(y_eval - self.train_mean_ * r / 10.0)))
                self.curve_.append(loss)
                if loss < best:
                    best, best_round = loss, r
                elif r - best_round >= self.patience:
                    break
            self.rounds_trained_ = len(self.curve_)
        self.rounds_used_ = self.rounds_trained_
        return self

    def predict(self, X):
        """Predict ``train_mean * rounds_used / 10`` for every row.

        Parameters
        ----------
        X : array-like
            Features; only their row count is used.

        Returns
        -------
        np.ndarray
            The constant prediction.

        """
        out = np.full((len(X), self._ncols), self.train_mean_ * self.rounds_used_ / 10.0)
        return out.ravel() if self._ncols == 1 else out


class CurveEarlyStoppingAdapter(BaseEarlyStoppingAdapter):
    """Adapter for `CurveRegressor`, recording every call it receives."""

    def __init__(self):
        self.calls = []

    def supports(self, estimator):
        """Whether ``estimator`` is a `CurveRegressor`.

        Parameters
        ----------
        estimator : object
            The estimator to check.

        Returns
        -------
        bool
            ``True`` for a `CurveRegressor`.

        """
        return isinstance(estimator, CurveRegressor)

    def validate(self, estimator, fit_params=None):
        """Record the call; every configuration is accepted.

        Parameters
        ----------
        estimator : CurveRegressor
            The estimator to validate.
        fit_params : dict or None, default=None
            The routed fit parameters.

        """
        self.calls.append(("validate", estimator.get_params(), fit_params))

    def prepare_fold_fit(self, estimator):
        """Return the estimator unchanged and no extra fit parameters.

        Parameters
        ----------
        estimator : CurveRegressor
            The estimator a fold will fit.

        Returns
        -------
        tuple
            A clone of ``estimator`` and an empty dict.

        """
        self.calls.append(("prepare_fold_fit", estimator.get_params()))
        return clone(estimator), {}

    def stopping_curve(self, fitted):
        """Return the fitted curve; lower is better.

        Parameters
        ----------
        fitted : CurveRegressor
            A fitted estimator.

        Returns
        -------
        tuple
            The curve and ``False``.

        """
        return np.asarray(fitted.curve_, dtype=float), False

    def truncate(self, fitted, n_rounds):
        """Predict with the first ``n_rounds`` rounds.

        Parameters
        ----------
        fitted : CurveRegressor
            A fitted estimator.
        n_rounds : int
            The round to cut to.

        """
        self.calls.append(("truncate", n_rounds))
        fitted.rounds_used_ = n_rounds

    def prepare_refit(self, estimator, n_rounds):
        """Configure the refit to train ``n_rounds`` rounds.

        Parameters
        ----------
        estimator : CurveRegressor
            The estimator the refit will fit.
        n_rounds : int
            The largest chosen round.

        Returns
        -------
        CurveRegressor
            A clone with ``n_rounds`` as its ceiling.

        """
        self.calls.append(("prepare_refit", n_rounds))
        return clone(estimator).set_params(n_rounds=n_rounds)
