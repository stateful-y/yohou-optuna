"""Module docstring."""

from collections.abc import Callable

import polars as pl
from yohou.utils import validate_scorer_data

from .base import BaseIntervalScorer


class MyIntervalMetric(BaseIntervalScorer):
    """Interval forecast metric.

    Evaluates prediction intervals (e.g., coverage, sharpness, calibration).

    Parameters
    ----------
    aggregation_method : list of str or str, default="all"
        Same as point scorers.
    panel_group_names : list of str or None, default=None
        Same as point scorers.
    component_names : list of str or None, default=None
        Same as point scorers.
    panel_group_weight : dict or None, default=None
        Same as point scorers.

    Examples
    --------
    >>> import polars as pl
    >>> from datetime import datetime
    >>> y_true = pl.DataFrame({"time": [datetime(2020, 1, 1), datetime(2020, 1, 2)], "value": [10.0, 20.0]})
    >>> y_pred = pl.DataFrame({
    ...     "observed_time": [datetime(2019, 12, 31)] * 2,
    ...     "time": [datetime(2020, 1, 1), datetime(2020, 1, 2)],
    ...     "value_lower_0.9": [8.0, 17.0],
    ...     "value_upper_0.9": [14.0, 23.0],
    ... })
    >>> metric = MyIntervalMetric()
    >>> score = metric.score(y_true, y_pred)
    >>> isinstance(score, float)
    True
    """

    def score(
        self,
        y_truth: pl.DataFrame,
        y_pred: pl.DataFrame,
        /,
        time_weight: Callable | pl.DataFrame | None = None,
        **params,
    ) -> float | pl.DataFrame:
        """Compute interval metric score.

        Parameters
        ----------
        y_truth : pl.DataFrame
            Ground truth with "time" and target columns.
        y_pred : pl.DataFrame
            Interval predictions with "time", "observed_time", and columns like:
            - "target_lower_0.9", "target_upper_0.9" (90% interval)
        time_weight : callable or pl.DataFrame, optional
            Time-based weights.
        **params : dict
            Metadata routing.

        Returns
        -------
        float or pl.DataFrame
            Metric score.

        """
        y_truth, y_pred = validate_scorer_data(self, y_truth=y_truth, y_pred=y_pred)

        scores = self._compute_scores(y_truth, y_pred)

        return self._aggregate_scores(
            scores=scores,
            y_truth=y_truth,
            y_pred=y_pred,
            time_weight=time_weight,
        )

    def _compute_scores(self, y_truth, y_pred):
        """Compute per-timestep, per-component interval scores.

        Parameters
        ----------
        y_truth : pl.DataFrame
            Ground truth values.
        y_pred : pl.DataFrame
            Interval predictions.

        Returns
        -------
        pl.DataFrame
            Scores with "time" column and score columns.

        """
        scores = pl.DataFrame({"time": y_pred["time"]})

        for col in self.component_names_:
            lower_cols = [c for c in y_pred.columns if c.startswith(f"{col}_lower_")]
            upper_cols = [c for c in y_pred.columns if c.startswith(f"{col}_upper_")]

            for lower_col, upper_col in zip(lower_cols, upper_cols, strict=False):
                coverage_level = lower_col.split("_")[-1]
                in_interval = ((y_truth[col] >= y_pred[lower_col]) & (y_truth[col] <= y_pred[upper_col])).cast(
                    pl.Float64
                )
                scores = scores.with_columns(in_interval.alias(f"{col}_coverage_{coverage_level}"))

        return scores
