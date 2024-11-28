"""Tests for the datasets module."""

# pylint: disable=unused-argument

from pathlib import Path
import unittest

import pandas as pd

from src.ml.datasets import ISEM2020Dataset


class TestDataset(unittest.TestCase):
    """Tests for the ISEM2020Dataset class."""

    data_dir = Path('tests', 'data')
    data_file = 'isem2020.csv'

    isem2020_train_set_start_time: str = '2019-12-31 23:00:00'
    isem2020_train_set_end_time: str = '2020-10-19 23:00:00'
    isem2020_test_set_start_time: str = '2020-10-20 00:00:00'
    isem2020_test_set_end_time: str = '2020-12-31 22:00:00'

    @classmethod
    def setUpClass(cls):
        """Create a dataset for testing purposes."""
        # Create data directory
        cls.data_dir.mkdir(parents=True, exist_ok=True)
        # Create a dataframe with a date range
        cls.csv_path = cls.data_dir / cls.data_file
        dates = pd.date_range(
            start=cls.isem2020_train_set_start_time,
            end=cls.isem2020_test_set_end_time,
            freq='H',
        )
        data = {'applicable_date': dates, 'value': range(len(dates))}
        df = pd.DataFrame(data)
        cls.df = df
        # Save the dataframe to a CSV file
        df.to_csv(cls.csv_path, index=False)

    @classmethod
    def tearDownClass(cls):
        """Remove the dataset created for testing purposes."""
        cls.csv_path.unlink()

    def test_dataset_initialization_with_path(self):
        """Test isem2020 dataset initialization with path to CSV file."""
        dataset = ISEM2020Dataset(path=self.csv_path)
        self.assertEqual(dataset.path, self.csv_path)
        self.assertIsNotNone(dataset.df)

    def test_dataset_initialization_without_path_or_url(self):
        """Test isem2020 without path or url raises ValueError."""
        with self.assertRaises(ValueError):
            ISEM2020Dataset()

    def test_split_data(self):
        """Test split_data method of ISEM2020Dataset class."""
        dataset = ISEM2020Dataset(path=self.csv_path)
        train_df, test_df = dataset.split_data()

        self.assertGreater(len(train_df), 0)
        self.assertEqual(str(train_df.head(1).index[0]), self.isem2020_train_set_start_time)
        self.assertEqual(str(train_df.tail(1).index[0]), self.isem2020_train_set_end_time)

        self.assertGreater(len(test_df), 0)
        self.assertEqual(str(test_df.head(1).index[0]), self.isem2020_test_set_start_time)
        self.assertEqual(str(test_df.tail(1).index[0]), self.isem2020_test_set_end_time)

        self.assertEqual(len(train_df) + len(test_df), len(dataset.df))

    def test_ensure_data(self):
        """Test ensure_data method of ISEM2020Dataset class."""
        dataset = ISEM2020Dataset(path=self.csv_path)
        dataset.ensure_data()
        self.assertIsNotNone(dataset.df)

    def test_ensure_data_empty(self):
        """Test ensure_data method of ISEM2020Dataset class with empty dataframe."""
        dataset = ISEM2020Dataset(path=self.csv_path)
        dataset.df = pd.DataFrame()
        with self.assertRaises(ValueError):
            dataset.ensure_data()
