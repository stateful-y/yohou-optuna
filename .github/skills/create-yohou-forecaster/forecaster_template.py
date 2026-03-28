"""Module docstring."""

import numbers

import polars as pl
from pydantic import StrictInt
from sklearn.base import _fit_context
from sklearn.utils._param_validation import Interval
from yohou.base import BaseTransformer

from .base import BasePointForecaster


class MyForecaster(BasePointForecaster):
    """NumPy-style docstring required.

    Parameters
    ----------
    param1 : int
        Description.
    target_transformer : BaseTransformer, optional
        Transformer for target variable (applied before forecasting).
    feature_transformer : BaseTransformer, optional
        Transformer for exogenous features X (applied before forecasting).

    Attributes
    ----------
    fitted_attr_ : type
        Description of fitted attribute (trailing underscore required).

    Examples
    --------
    >>> import polars as pl
    >>> from datetime import datetime
    >>> time = pl.datetime_range(
    ...     start=datetime(2020, 1, 1), end=datetime(2020, 2, 1), interval="1d", eager=True
    ... )
    >>> y = pl.DataFrame({"time": time, "value": range(len(time))})
    >>> forecaster = MyForecaster(param1=10)
    >>> forecaster.fit(y, forecasting_horizon=5)
    MyForecaster(param1=10)
    >>> y_pred = forecaster.predict(forecasting_horizon=5)
    >>> len(y_pred)
    5
    """

    _parameter_constraints: dict = {
        **BasePointForecaster._parameter_constraints,
        "param1": [Interval(numbers.Integral, 1, None, closed="left")],
    }

    def __init__(
        self,
        param1: int,
        target_transformer: BaseTransformer | None = None,
        feature_transformer: BaseTransformer | None = None,
    ):
        super().__init__(
            target_transformer=target_transformer,
            feature_transformer=feature_transformer,
        )
        self.param1 = param1
        # DO NOT validate parameters here — validation happens at fit time via @_fit_context

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(
        self,
        y: pl.DataFrame,
        X: pl.DataFrame | None = None,
        forecasting_horizon: StrictInt = 1,
        **params,
    ) -> "MyForecaster":
        """Fit forecaster.

        Parameters
        ----------
        y : pl.DataFrame
            Target time series with "time" column.
        X : pl.DataFrame, optional
            Exogenous features with "time" column.
        forecasting_horizon : int, default=1
            Number of steps ahead to forecast.
        **params : dict
            Metadata routing parameters.

        Returns
        -------
        self

        """
        y_t, X_t = self._pre_fit(y=y, X=X, forecasting_horizon=forecasting_horizon)
        # Your fitting logic using y_t, X_t (already transformed)
        # Must set at least one fitted attribute with trailing underscore
        self.fitted_attr_ = ...  # Example: self.model_, self.coefficients_, etc.
        return self

    def predict(
        self,
        X: pl.DataFrame | None = None,
        forecasting_horizon: StrictInt | None = None,
        panel_group_names: list[str] | None = None,
        predict_transformed: bool = False,
        **params,
    ) -> pl.DataFrame:
        """Generate forecasts.

        Parameters
        ----------
        X : pl.DataFrame, optional
            Exogenous features for forecast period (must have "time" column).
        forecasting_horizon : int, optional
            Number of steps ahead to forecast. If None, uses value from fit().
        panel_group_names : list of str, optional
            Panel group prefixes to predict (for panel data).
        predict_transformed : bool, default=False
            If True, return predictions in transformed space.
        **params : dict
            Metadata routing parameters.

        Returns
        -------
        pl.DataFrame
            Predictions with "observed_time", "time", and target columns.

        """
        if forecasting_horizon is None:
            forecasting_horizon = self._forecasting_horizon
        # Your prediction logic
        y_pred = ...  # Must be pl.DataFrame with target columns (no time yet)
        return self._add_time_columns(y_pred)  # CRITICAL: adds observed_time + time
