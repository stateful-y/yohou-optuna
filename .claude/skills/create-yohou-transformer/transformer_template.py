"""Module docstring."""

import numbers

import polars as pl
from sklearn.base import _fit_context
from sklearn.utils._param_validation import Interval
from sklearn.utils.validation import check_is_fitted
from yohou.base import BaseTransformer
from yohou.utils import validate_transformer_data


class MyTransformer(BaseTransformer):
    """NumPy-style docstring required.

    Parameters
    ----------
    param1 : int
        Description.

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
    >>> X = pl.DataFrame({"time": time, "value": range(len(time))})
    >>> transformer = MyTransformer(param1=10)
    >>> transformer.fit(X)
    MyTransformer(param1=10)
    >>> X_t = transformer.transform(X)
    >>> "time" in X_t.columns
    True
    """

    _parameter_constraints: dict = {
        **BaseTransformer._parameter_constraints,
        "param1": [Interval(numbers.Integral, 1, None, closed="left")],
    }

    def __init__(self, param1: int):
        self.param1 = param1
        # DO NOT call super().__init__() — BaseTransformer has no __init__
        # DO NOT validate parameters here — validation happens at fit time

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X: pl.DataFrame, y: pl.DataFrame | None = None, **params) -> "MyTransformer":
        """Fit transformer.

        Parameters
        ----------
        X : pl.DataFrame
            Feature time series with "time" column.
        y : pl.DataFrame, optional
            Target time series (for API compatibility, often unused).
        **params : dict
            Metadata routing parameters.

        Returns
        -------
        self

        """
        # Validate input data
        X = validate_transformer_data(self, X=X, reset=True)

        # OPTIONAL: Set observation horizon for stateful transformers
        # self._observation_horizon = 10  # Keep last 10 observations

        # Call parent fit (stores schema, memory, etc.)
        BaseTransformer.fit(self, X, y, **params)

        # Your fitting logic
        self.fitted_attr_ = ...  # Must set at least one fitted attribute

        return self

    def transform(self, X: pl.DataFrame, **params) -> pl.DataFrame:
        """Transform input time series.

        Parameters
        ----------
        X : pl.DataFrame
            Feature time series with "time" column.
        **params : dict
            Metadata routing parameters.

        Returns
        -------
        pl.DataFrame
            Transformed time series (MUST include "time" column).

        """
        check_is_fitted(self, ["X_schema_", "feature_names_in_", "n_features_in_"])
        X = validate_transformer_data(self, X=X, reset=False, check_continuity=False)

        # Your transformation logic
        X_t = ...  # Transform X (must preserve "time" column)

        return X_t

    def inverse_transform(self, X: pl.DataFrame, X_p: pl.DataFrame | None = None, **params) -> pl.DataFrame:
        """Inverse transform (optional, only if transformation is reversible).

        Parameters
        ----------
        X : pl.DataFrame
            Transformed time series with "time" column.
        X_p : pl.DataFrame or None, default=None
            Previous observations (needed for stateful transformers).
        **params : dict
            Metadata routing parameters.

        Returns
        -------
        pl.DataFrame
            Original scale time series.

        """
        check_is_fitted(self, ["fitted_attr_"])
        # ALWAYS validate input data in inverse_transform
        # For stateless transformers:
        X = validate_transformer_data(self, X=X, reset=False, check_continuity=False)
        # For stateful transformers:
        # X_t, X_p = validate_transformer_data(
        #     self, X=X, reset=False, inverse=True, X_p=X_p,
        #     observation_horizon=self.observation_horizon, stateful=True
        # )
        # Your inverse transformation logic
        X_inv = ...
        return X_inv
