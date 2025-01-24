"""Functions for setting up and tearing down tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_openml

from ml.logs import logger


def download_bike_sharing(
    row_limit: int | None = None,
    csv_path: str | Path | None = None,
) -> pd.DataFrame:
    """Download bike sharing dataset.

    :param int | None row_limit: Number of rows to download, or None for all rows.
    :param str | Path | None csv_path: Path to save the dataset as a CSV file, or None to not save.
    :return pd.DataFrame df: Bike sharing dataset.
    """
    # Try to download the dataset
    df = attempt_download('Bike_Sharing_Demand', 2)

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


def attempt_download(dataset_name: str, version: int) -> pd.DataFrame:
    """Attempt to download a dataset from OpenML.

    :param str dataset_name: Name of the dataset.
    :param int version: Version of the dataset.
    :return pd.DataFrame df: Dataset.
    """
    # data_home = os.path.join('tests', 'data', 'openml')
    kwargs = {
        'name': dataset_name,
        'version': version,
        'as_frame': True,
        # 'data_home': data_home,
    }

    data = None

    try:
        data = fetch_openml(**kwargs)
    except Exception:
        logger.critical('Failed A')

    try:
        if data is None:
            data = fetch_openml(**kwargs, cache=False)
    except Exception:
        logger.critical('Failed B')

    try:
        if data is None:
            data = fetch_openml(**kwargs, parser='auto')
    except Exception:
        logger.critical('Failed C')

    # shutil.rmtree(data_home)
    # Try to download the dataset
    # try:
    #     data = fetch_openml(**kwargs, parser='auto')
    # except ValueError:
    #     data = fetch_openml(**kwargs, cache=False)
    #     # # Checksum error
    #     # if 'md5 checksum' in str(e):
    #     #     # Delete the cached dataset
    #     #     # shutil.rmtree(data_home)
    #     #     # Try to download the dataset again
    #     #     data = fetch_openml(**kwargs, cache=False)
    #     #     data = fetch_openml(**kwargs, cache=False)
    #     # else:
    #     #     raise e
    # except TypeError:
    #     # Newer versions of scikit-learn
    #     data = fetch_openml(**kwargs)
    # Get the dataset as a pandas DataFrame
    df = data.frame
    return df
