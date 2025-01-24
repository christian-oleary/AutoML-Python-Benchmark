"""Tests for univariate forecasting."""

import os
from pathlib import Path

import pandas as pd
import pytest
from sklearn.experimental import enable_iterative_imputer  # pylint: disable=W0611  # noqa: F401

from tests.conftest import download_bike_sharing

# from ml.forecasting import Forecasting
from ml.validation import Library, Task


class TestDataset:
    """Test dataset methods."""

    @pytest.fixture(autouse=True)
    def setup(self, overwrite=True):
        """Setup for functional tests."""

        class Config:  # pylint: disable=R0903
            """Default testing configuration."""

            data_dir = os.path.join('tests', 'data', 'univariate')
            libraries = ['test']
            nproc = 1
            repeat_results = True
            results_dir = None
            task = Task.UNIVARIATE_FORECASTING.value
            time_limit = 15

        config = Config()

        # Create test dataset for forecasting
        os.makedirs(config.data_dir, exist_ok=True)
        forecasting_test_path = Path(config.data_dir, 'forecasting_data.csv')

        if overwrite or not forecasting_test_path.exists():
            # Download bike sharing dataset
            df = download_bike_sharing(200, forecasting_test_path)
            # Create metadata DataFrame
            metadata = {
                'horizon': 6,
                'has_nans': False,
                'equal_length': False,
                'num_rows': df.shape[0],
                'num_cols': df.shape[1],
            }
            metadata['frequency'] = 48
            metadata['file'] = 'forecasting_data.csv'
            metadata = pd.DataFrame([metadata])
            # Save metadata
            metadata_path = Path(config.data_dir, '0_metadata.csv')
            metadata.to_csv(metadata_path, index=False)

        return config

    def test_forecasting_libraries(self, setup):  # pylint: disable=unused-argument
        """Test forecasting on a small dataset."""
        forecasters = Library.get_options()
        if len(forecasters) == 0:
            raise ValueError('No forecasters found')
        # Forecasting().run_forecasting_libraries(setup)
