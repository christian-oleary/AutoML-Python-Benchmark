"""Functions for setting up and tearing down tests."""

from __future__ import annotations
from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_openml

from src.ml.logs import logger


def download_bike_sharing(
    row_limit: int | None = None,
    csv_path: str | Path | None = None,
) -> pd.DataFrame:
    """Download bike sharing dataset.

    :param int | None row_limit: Number of rows to download, or None for all rows.
    :param str | Path | None csv_path: Path to save the dataset as a CSV file, or None to not save.
    :return pd.DataFrame df: Bike sharing dataset.
    """

    def download(**kwargs):
        return fetch_openml('Bike_Sharing_Demand', version=2, as_frame=True, **kwargs)

    # Try to download the dataset
    try:
        bike_sharing = bike_sharing = download(parser='auto')
    # Checksum error
    except ValueError as error:
        text = error.args[0]
        if 'checksum' in text:
            logger.warning(text)
            bike_sharing = download(cache=False)
        else:
            raise error
    # Newer versions of scikit-learn
    except TypeError:
        bike_sharing = download()

    # Get the dataset as a pandas DataFrame
    df = bike_sharing.frame

    # Limit the number of rows
    if row_limit is not None:
        df = df.head(row_limit)

    # Add timestamps
    timestamps = pd.date_range(start='2012-1-1 00:00:00', periods=len(df), freq='30T')
    df['applicable_date'] = timestamps

    # Filter columns
    # df = df[['temp', 'feel_temp', 'humidity', 'windspeed']]
    df = df[['applicable_date', 'temp']]

    # Save the dataset
    if csv_path is not None:
        df.to_csv(csv_path, index=False)
    return df
