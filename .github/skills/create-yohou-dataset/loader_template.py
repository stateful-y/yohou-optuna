"""Loader function template for a new dataset."""

import importlib.resources

import polars as pl
from yohou.datasets import data


def _load_dataset(name: str) -> pl.DataFrame:
    """Load a dataset by name from the bundled data directory.

    Parameters
    ----------
    name : str
        Name of the dataset (without .parquet extension).

    Returns
    -------
    pl.DataFrame
        The loaded dataset.

    """
    with importlib.resources.as_file(importlib.resources.files(data).joinpath(f"{name}.parquet")) as p:
        return pl.read_parquet(p)


def load_my_dataset() -> pl.DataFrame:
    """Load the MyDataset time series dataset.

    This dataset contains <brief description of what the data represents>.

    Returns
    -------
    pl.DataFrame
        DataFrame with columns:
        - ``time`` : datetime — Timestamps at <interval> frequency.
        - ``value`` : float — <Description of column>.
        - ``feature_1`` : float — <Description of feature>.

    Notes
    -----
    - **Source**: <URL or citation>
    - **License**: <License type, e.g., CC BY 4.0>
    - **Time range**: <start> to <end>
    - **Frequency**: <e.g., hourly, daily, monthly>
    - **Number of observations**: <count>
    - **Missing values**: <None / description>

    References
    ----------
    .. [1] Author, "Title", Year. URL

    Examples
    --------
    >>> from yohou.datasets import load_my_dataset
    >>> df = load_my_dataset()
    >>> df.shape  # doctest: +SKIP
    (100, 2)
    >>> df.columns
    ['time', 'value']
    """
    return _load_dataset("my_dataset")
