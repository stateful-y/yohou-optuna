"""Module docstring."""

from collections.abc import Callable

import polars as pl
from yohou.utils import validate_scorer_data

from .base import BasePointScorer


class MyMetric(BasePointScorer):
    """NumPy-style docstring required.

    Parameters
    ----------
    aggregation_method : list of str or str, default="all"
        Dimensions to aggregate over. Options:
        - "timewise": Aggregate across time, return per-component DataFrame
        - "componentwise": Aggregate across components, return per-timestep DataFrame
        - "groupwise": Aggregate across panel groups (panel data only)
        - "all": Aggregate across all dimensions (returns scalar)
    panel_group_names : list of str or None, default=None
        List of panel group names to include in scoring.
    component_names : list of str or None, default=None
        List of component (target column) names to include.
    panel_group_weight : dict or None, default=None
        Dictionary mapping panel group names to weights.

    Attributes
    ----------
    lower_is_better : bool
        True if lower scores are better (e.g., MAE, RMSE).

    Examples
    --------
    >>> import polars as pl
    >>> from datetime import datetime
    >>> y_true = pl.DataFrame({
    ...     "time": [datetime(2020, 1, 1), datetime(2020, 1, 2), datetime(2020, 1, 3)],
    ...     "value": [10.0, 20.0, 30.0],
    ... })
    >>> y_pred = pl.DataFrame({
    ...     "observed_time": [datetime(2019, 12, 31)] * 3,
    ...     "time": [datetime(2020, 1, 1), datetime(2020, 1, 2), datetime(2020, 1, 3)],
    ...     "value": [12.0, 19.0, 28.0],
    ... })
    >>> metric = MyMetric()
    >>> score = metric.score(y_true, y_pred)
    >>> isinstance(score, float)
    True
    """

    _parameter_constraints: dict = {
        **BasePointScorer._parameter_constraints,
        # Add metric-specific parameters here
    }

    def __init__(
        self,
        aggregation_method: list[str] | str = "all",
        panel_group_names: list[str] | None = None,
        component_names: list[str] | None = None,
        panel_group_weight: dict[str, float] | None = None,
    ):
        super().__init__(
            aggregation_method=aggregation_method,
            panel_group_names=panel_group_names,
            component_names=component_names,
            panel_group_weight=panel_group_weight,
        )

    def score(
        self,
        y_truth: pl.DataFrame,
        y_pred: pl.DataFrame,
        /,
        time_weight: Callable | pl.DataFrame | None = None,
        **params,
    ) -> float | pl.DataFrame:
        """Compute metric score.

        Parameters
        ----------
        y_truth : pl.DataFrame
            Ground truth values with "time" column.
        y_pred : pl.DataFrame
            Predicted values with "observed_time" and "time" columns.
        time_weight : callable or pl.DataFrame, optional
            Time-based weights for scoring.
        **params : dict
            Metadata routing parameters.

        Returns
        -------
        float or pl.DataFrame
            Scalar score (if aggregation_method="all") or DataFrame with partial aggregations.

        """
        # Validate inputs
        y_truth, y_pred = validate_scorer_data(
            self,
            y_truth=y_truth,
            y_pred=y_pred,
        )

        # Compute per-timestep, per-component scores
        scores = self._compute_scores(y_truth, y_pred)

        # Apply time weighting and aggregate
        return self._aggregate_scores(
            scores=scores,
            y_truth=y_truth,
            y_pred=y_pred,
            time_weight=time_weight,
        )

    def _compute_scores(
        self,
        y_truth: pl.DataFrame,
        y_pred: pl.DataFrame,
    ) -> pl.DataFrame:
        """Compute per-timestep, per-component scores (before aggregation).

        Parameters
        ----------
        y_truth : pl.DataFrame
            Ground truth values (already validated).
        y_pred : pl.DataFrame
            Predicted values (already validated).

        Returns
        -------
        pl.DataFrame
            Scores with "time" column and score columns for each component.

        """
        # Extract target columns (exclude time columns)
        target_cols = [c for c in y_truth.columns if c != "time"]

        # Compute element-wise metric
        # Example: Absolute error
        scores = pl.DataFrame({"time": y_pred["time"]})
        for col in target_cols:
            scores = scores.with_columns((y_truth[col] - y_pred[col]).abs().alias(col))

        return scores
