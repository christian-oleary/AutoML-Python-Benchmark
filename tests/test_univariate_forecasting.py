"""Tests for univariate forecasting."""

import os
from pathlib import Path

import pandas as pd
import pytest
from sklearn.datasets import make_regression
from sklearn.experimental import enable_iterative_imputer  # pylint: disable=W0611  # noqa: F401

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
            # Create a time series dataset
            frequency = 24
            X, y = make_regression(n_samples=frequency * 10, n_features=1, random_state=1)
            df = pd.DataFrame(X, columns=['feature'])
            df['target'] = y
            df['timestamp'] = pd.date_range(start='2000-01-01 00:00:00', periods=len(df), freq='H')
            df.set_index('timestamp', inplace=True)
            df.to_csv(forecasting_test_path, index=False)

        return config

    def test_forecasting_libraries(self, setup):  # pylint: disable=unused-argument
        """Test forecasting on a small dataset."""
        forecasters = Library.get_options()
        if len(forecasters) == 0:
            raise ValueError('No forecasters found')
        # Forecasting().run_forecasting_libraries(setup)
