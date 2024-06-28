"""Functional tests"""

import os

import pandas as pd
import pytest
from sklearn.datasets import fetch_openml
from sklearn.experimental import enable_iterative_imputer # pylint: disable=W0611  # noqa

from src.dataset_formatting import DatasetFormatter
from src.forecasting import Forecasting
from src.util import Utils
from src.validation import Library, Task


def setup(overwrite=True):
    """Setup for functional tests"""

    class Config:  # pylint: disable=R0903
        """Default testing configuration"""
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
    forecasting_test_path = os.path.join(config.data_dir, 'forecasting_data.csv')

    if overwrite or not os.path.exists(forecasting_test_path):
        try:
            bike_sharing = fetch_openml('Bike_Sharing_Demand', version=2, as_frame=True, parser='auto')
        except TypeError:  # Newer versions
            bike_sharing = fetch_openml('Bike_Sharing_Demand', version=2, as_frame=True)

        df = bike_sharing.frame
        df = df.head(200)
        df = df[['temp', 'feel_temp', 'humidity', 'windspeed']]
        timestamps = pd.date_range(start='2012-1-1 00:00:00', periods=len(df), freq='30T')

        df['applicable_date'] = timestamps
        df = df[['applicable_date', 'temp']]
        df.to_csv(forecasting_test_path, index=False)

        metadata_path = os.path.join(config.data_dir, '0_metadata.csv')
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
        metadata.to_csv(metadata_path, index=False)
    return config


setup(overwrite=True)


@pytest.fixture(autouse=True)
def fixture():
    """Basic fixture for tests"""
    config = setup(overwrite=False)
    yield config


def test_dataset_formatting_extract_forecasting_data(fixture):  # pylint: disable=W0613,W0621
    """Test zip file parsing and data extraction"""

    with pytest.raises(NotADirectoryError):
        DatasetFormatter().extract_forecasting_data('README.md')

    with pytest.raises(IOError):
        DatasetFormatter().extract_forecasting_data('tests')


# def test_dataset_formatting_get_global_forecasting_datasets(fixture):  # pylint: disable=W0621
#     """Test CSV file discovery"""

#     config = fixture
#     Utils.get_csv_datasets(config.univariate_forecasting_data_dir)
#     with pytest.raises(IOError):
#         DatasetFormatter().extract_forecasting_data('tests')


def test_forecasting_run_forecasting_libraries_test(fixture):  # pylint: disable=W0621
    """Test running a basic forecasting model on a small dataset"""

    config = fixture
    forecasters = Library.get_options()
    if len(forecasters) == 0:
        raise ValueError('No forecasters found')
    Forecasting().run_forecasting_libraries(config)
