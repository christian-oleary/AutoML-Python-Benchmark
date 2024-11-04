"""Tests for the datasets module."""

# pylint: disable=unused-argument

from pathlib import Path

import argparse
import pandas as pd
import pytest

from src.automl.datasets import ISEM2020Dataset, DatasetFormatter
from src.automl.validation import Task


class TestDataset:
    """Tests for the ISEM2020Dataset class."""

    csv_path = Path('tests', 'data', 'isem2020.csv')

    isem2020_train_set_start_time: str = '2019-12-31 23:00:00'
    isem2020_train_set_end_time: str = '2020-10-19 23:00:00'
    isem2020_test_set_start_time: str = '2020-10-20 00:00:00'
    isem2020_test_set_end_time: str = '2020-12-31 22:00:00'

    @pytest.fixture(autouse=True)
    def create_dataset(self):
        """Create a dataset for testing purposes."""
        dates = pd.date_range(
            start=TestDataset.isem2020_train_set_start_time,
            end=TestDataset.isem2020_test_set_end_time,
            freq='H',
        )
        data = {'applicable_date': dates, 'value': range(len(dates))}
        df = pd.DataFrame(data)
        df.to_csv(TestDataset.csv_path, index=False)
        return df

    def test_dataset_initialization_with_path(self, create_dataset):
        """Test isem2020 dataset initialization with path to CSV file."""
        dataset = ISEM2020Dataset(path=TestDataset.csv_path)
        assert dataset.path == TestDataset.csv_path
        assert dataset.df is not None

    def test_dataset_initialization_without_path_or_url(self):
        """Test isem2020 without path or url raises ValueError."""
        with pytest.raises(ValueError):
            ISEM2020Dataset()

    def test_split_data(self, create_dataset):
        """Test split_data method of ISEM2020Dataset class."""
        dataset = ISEM2020Dataset(path=TestDataset.csv_path)
        train_df, test_df = dataset.split_data()

        assert len(train_df) > 0
        assert str(train_df.head(1).index[0]) == TestDataset.isem2020_train_set_start_time
        assert str(train_df.tail(1).index[0]) == TestDataset.isem2020_train_set_end_time

        assert len(test_df) > 0
        assert str(test_df.head(1).index[0]) == TestDataset.isem2020_test_set_start_time
        assert str(test_df.tail(1).index[0]) == TestDataset.isem2020_test_set_end_time

        assert len(train_df) + len(test_df) == len(dataset.df)

    def test_ensure_data(self, create_dataset):
        """Test ensure_data method of ISEM2020Dataset class."""
        dataset = ISEM2020Dataset(path=TestDataset.csv_path)
        dataset.ensure_data()
        assert dataset.df is not None

    def test_ensure_data_empty(self, create_dataset):
        """Test ensure_data method of ISEM2020Dataset class with empty dataframe."""
        dataset = ISEM2020Dataset(path=TestDataset.csv_path)
        dataset.df = pd.DataFrame()
        with pytest.raises(ValueError):
            dataset.ensure_data()


class TestDatasetFormatter:
    """Tests for the DatasetFormatter class."""

    data_dir = Path('tests', 'data')
    csv_path = Path(data_dir, 'data.csv')
    tsf_path = Path(data_dir, 'data.tsf')
    zip_path = Path(data_dir, 'data.zip')
    metadata_path = Path(data_dir, '0_metadata.csv')

    def test_format_univariate_forecasting_data(self, tmp_path):
        """Test format_univariate_forecasting_data method of DatasetFormatter class."""
        # Create a temporary CSV file
        data = {
            'date': pd.date_range(start='1/1/2020', periods=100, freq='H'),
            'value': range(100)
        }
        df = pd.DataFrame(data)
        df.to_csv(TestDatasetFormatter.csv_path, index=False)

        class TestConfig(argparse.Namespace):
            """Configuration class for testing purposes."""
            task = Task.UNIVARIATE_FORECASTING.value
            data_dir = tmp_path

        formatter = DatasetFormatter()
        formatter.format_data(TestConfig())

    # def test_format_global_forecasting_data(self):
    #     """Test format_global_forecasting_data method of DatasetFormatter class."""
    #     formatter = DatasetFormatter()

    #     # Create .tsf file (time series file)
    #     time_series = 'series_name,start_timestamp,value\n'  # Header
    #     time_series += 'series1,2020-01-01 00-00-00,1 2 3 4 5'  # Data
    #     with open(TestDatasetFormatter.tsf_path, 'w', encoding='utf-8') as f:
    #         f.write(time_series)

    #     # Create .zip file
    #     with zipfile.ZipFile(TestDatasetFormatter.zip_path, 'w') as f:
    #         f.write(TestDatasetFormatter.tsf_path, arcname="test.tsf")

    #     # Format global forecasting data
    #     formatter.format_global_forecasting_data(
    #         TestDatasetFormatter.data_dir, gather_metadata=True
    #     )
    #     assert TestDatasetFormatter.metadata_path.exists()
