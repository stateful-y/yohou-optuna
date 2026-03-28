"""Module docstring."""

import numbers
from collections.abc import Iterator
from typing import Any

import numpy as np
import polars as pl
from sklearn.utils._param_validation import Interval
from yohou.model_selection import BaseSplitter
from yohou.utils import validate_splitter_data


class MySplitter(BaseSplitter):
    """NumPy-style docstring required.

    Parameters
    ----------
    n_splits : int, default=3
        Number of splits. Must be at least 2.
    test_size : int, optional
        Size of test set for each split.

    Examples
    --------
    >>> import polars as pl
    >>> from datetime import datetime, timedelta
    >>> time = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(100)]
    >>> y = pl.DataFrame({"time": time, "value": range(100)})
    >>> splitter = MySplitter(n_splits=3, test_size=10)
    >>> splits = list(splitter.split(y))
    >>> len(splits)
    3
    >>> train, test = splits[0]
    >>> len(test)
    10
    """

    _parameter_constraints: dict = {
        **BaseSplitter._parameter_constraints,
        "n_splits": [Interval(numbers.Integral, 2, None, closed="left")],
        "test_size": [Interval(numbers.Integral, 1, None, closed="left"), None],
    }

    def __init__(self, n_splits: int = 3, test_size: int | None = None):
        self.n_splits = n_splits
        self.test_size = test_size

    def split(
        self,
        y: pl.DataFrame,
        X: pl.DataFrame | None = None,
    ) -> Iterator[tuple[np.ndarray[Any, np.dtype[np.intp]], np.ndarray[Any, np.dtype[np.intp]]]]:
        """Generate indices to split time series data.

        Parameters
        ----------
        y : pl.DataFrame
            Target time series with mandatory "time" column.
        X : pl.DataFrame, optional
            Exogenous features (for signature compatibility, not used in splitting logic).

        Yields
        ------
        train : ndarray
            Training set row indices for that split.
        test : ndarray
            Test set row indices for that split.

        """
        # Validate data
        y = validate_splitter_data(self, y=y, X=X)
        _n_samples = len(y)  # noqa: F841

        # Generate test indices
        for test_indices in self._iter_test_indices(y, X):
            # Compute train indices (all indices before test start)
            train_indices = np.arange(0, test_indices[0])
            yield train_indices, test_indices

    def _iter_test_indices(
        self,
        y: pl.DataFrame,
        X: pl.DataFrame | None = None,
    ) -> Iterator[np.ndarray[Any, np.dtype[np.intp]]]:
        """Generate test indices for each split.

        Parameters
        ----------
        y : pl.DataFrame
            Target time series.
        X : pl.DataFrame, optional
            Exogenous features.

        Yields
        ------
        test : ndarray
            Test set indices for this split.

        """
        n_samples = len(y)
        test_size = self.test_size or n_samples // (self.n_splits + 1)

        # Generate test indices for each split
        for i in range(self.n_splits):
            test_start = n_samples - (self.n_splits - i) * test_size
            test_end = test_start + test_size
            yield np.arange(test_start, test_end)

    def get_n_splits(
        self,
        y: pl.DataFrame | None = None,
        X: pl.DataFrame | None = None,
    ) -> int:
        """Returns the number of splitting iterations in the cross-validator.

        Parameters
        ----------
        y : pl.DataFrame, optional
            Target time series.
        X : pl.DataFrame, optional
            Exogenous features.

        Returns
        -------
        n_splits : int
            Number of splits.

        """
        return self.n_splits
