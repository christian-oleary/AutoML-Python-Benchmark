"""Functional tests"""

import os

import pandas as pd
import pytest
from sklearn.datasets import fetch_openml
from sklearn.experimental import enable_iterative_imputer # pylint: disable=W0611

from src.dataset_formatting import DatasetFormatter
from src.forecasting import Forecasting
from src.util import Utils


def setup(overwrite=True):
    """Setup for functional tests"""

    class Config: # pylint: disable=R0903
        """Default testing configuration"""
        libraries = [ 'test' ]
        time_limit = 15
        univariate_forecasting_data_dir = os.path.join('tests', 'data', 'univariate')
        global_forecasting_data_dir = os.path.join('tests', 'data', 'global')
        repeat_results = True
        results_dir = None
        nproc = 1

    config = Config()

    # Create test dataset for forecasting
    forecasting_data_dir = os.path.join('tests', 'data')

    for forecast_type in ['global', 'univariate']:
        data_dir = os.path.join(forecasting_data_dir, forecast_type)
        os.makedirs(data_dir, exist_ok=True)
        forecasting_test_path = os.path.join(data_dir, 'forecasting_data.csv')

        if overwrite or not os.path.exists(forecasting_test_path):
            try:
                bike_sharing = fetch_openml('Bike_Sharing_Demand', version=2, as_frame=True, parser='auto')
            except TypeError as _: # Newer versions
                bike_sharing = fetch_openml('Bike_Sharing_Demand', version=2, as_frame=True)

            df = bike_sharing.frame
            df = df.head(200)
            df = df[['temp', 'feel_temp', 'humidity', 'windspeed']]
            timestamps = pd.date_range(start='2012-1-1 00:00:00', periods=len(df), freq='30T')

            if forecast_type == 'univariate':
                df['applicable_date'] = timestamps
                df = df[['applicable_date', 'temp']]
                df.to_csv(forecasting_test_path, index=False)
            else:
                df.index = timestamps
                df.to_csv(forecasting_test_path)

            metadata_path = os.path.join(data_dir, '0_metadata.csv')
            metadata = {
                'horizon': 6,
                'has_nans': False,
                'equal_length': False,
                'num_rows': df.shape[0],
                'num_cols': df.shape[1],
            }

            if forecast_type == 'univariate':
                metadata['frequency'] = 48
                metadata['file'] = 'forecasting_data.csv'
            else:
                metadata['frequency'] = 'half_hourly'
                metadata['file'] = 'forecasting_data.tsf'

            metadata = pd.DataFrame([metadata])
            metadata.to_csv(metadata_path, index=False)
    return config

setup(overwrite=True)

@pytest.fixture(autouse=True)
def fixture():
    """Basic fixture for tests"""
    config = setup(overwrite=False)
    yield config


def test_dataset_formatting_extract_forecasting_data(fixture): # pylint: disable=W0613,W0621
    """Test zip file parsing and data extraction"""

    with pytest.raises(NotADirectoryError):
        DatasetFormatter.extract_forecasting_data('README.md')

    with pytest.raises(IOError):
        DatasetFormatter.extract_forecasting_data('tests')


def test_dataset_formatting_get_globabl_forecasting_datasets(fixture): # pylint: disable=W0621
    """Test CSV file discovery"""

    config = fixture

    Utils.get_csv_datasets(config.univariate_forecasting_data_dir)
    with pytest.raises(IOError):
        DatasetFormatter.extract_forecasting_data('tests')


def test_forecasting_run_forecasting_libraries_test(fixture): # pylint: disable=W0621
    """Test running a basic forecasting model on a small dataset"""

    config = fixture

    forecasters = Forecasting().forecaster_names
    assert len(forecasters) > 0, 'No forecasters found'

    Forecasting().run_forecasting_libraries(config.univariate_forecasting_data_dir, config, 'univariate')
