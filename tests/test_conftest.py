"""Tests for conftest.py."""

import pandas as pd

from tests.conftest import download_bike_sharing

DATE_COL = 'applicable_date'
TEMP_COL = 'temp'


def _check_df(df):
    assert not df.empty
    assert DATE_COL in df.columns
    assert TEMP_COL in df.columns


def test_download_bike_sharing_all_rows():
    """Test downloading the bike sharing dataset with all rows."""
    df = download_bike_sharing()
    _check_df(df)


def test_download_bike_sharing_row_limit():
    """Test downloading the bike sharing dataset with a row limit."""
    df = download_bike_sharing(row_limit=3)
    assert len(df) == 3
    _check_df(df)


def test_download_bike_sharing_save_csv(tmp_path):
    """Test saving the bike sharing dataset as a CSV file.

    :param pathlib.Path tmp_path: Temporary directory.
    """
    csv_path = tmp_path / 'bike_sharing.csv'
    df = download_bike_sharing(csv_path=csv_path)
    _check_df(df)

    assert csv_path.exists()
    df_csv = pd.read_csv(csv_path)
    _check_df(df_csv)
    df_csv[DATE_COL] = pd.to_datetime(df_csv[DATE_COL])

    pd.testing.assert_frame_equal(df, df_csv)
