import os

import pandas as pd
import pytest
from sklearn.datasets import fetch_openml

from src.dataset_formatting import DatasetFormatting
from src.forecasting import Forecasting
from src.util import Utils


@pytest.fixture(autouse=True)
def fixture():
    """Basic fixture for tests"""

    # Create test dataset for forecasting
    forecasting_data_dir = os.path.join('tests', 'data', 'forecasting')
    forecasting_test_path = os.path.join(forecasting_data_dir, 'forecasting_data.csv')

    if not os.path.exists(forecasting_test_path):
        os.makedirs(forecasting_data_dir, exist_ok=True)

        bike_sharing = fetch_openml('Bike_Sharing_Demand', version=2, as_frame=True)
        df = bike_sharing.frame
        df = df.head(200)
        df = df[['temp', 'feel_temp', 'humidity', 'windspeed']]
        timestamps = pd.date_range(start='2012-1-1 00:00:00', periods=len(df), freq='30T')
        df.index = timestamps
        df.to_csv(forecasting_test_path)

    metadata_path = os.path.join(forecasting_data_dir, '0_metadata.csv')
    if not os.path.exists(metadata_path):
        metadata = {
            'file': 'forecasting_data.tsf',
            'frequency': 'half_hourly',
            'horizon': 6,
            'has_nans': False,
            'equal_length': False,
            'num_rows': 200,
            'num_cols': 4,
        }
        metadata = pd.DataFrame([metadata])
        metadata.to_csv(metadata_path, index=False)
    yield forecasting_data_dir, forecasting_test_path


def test_dataset_formatting_extract_forecasting_data(fixture):
    """Test zip file parsing and data extraction"""

    with pytest.raises(NotADirectoryError):
        DatasetFormatting.extract_forecasting_data('README.md')

    with pytest.raises(IOError):
        DatasetFormatting.extract_forecasting_data('tests')

    DatasetFormatting.extract_forecasting_data(os.path.join('data', 'forecasting'))


def test_forecasting_run_forecasting_libraries(fixture):
    """Test running forecasting models"""

    forecasting_data_dir, _ = fixture
    forecasters = Forecasting.get_forecaster_names()
    assert len(forecasters) > 0, 'No forecasters found'
    Forecasting.run_forecasting_libraries(forecasters, forecasting_data_dir)


def test_util_get_csv_datasets(fixture):
    """Test CSV file discovery"""

    forecasting_data_dir, _ = fixture

    Utils.get_csv_datasets(forecasting_data_dir)
    with pytest.raises(IOError):
        DatasetFormatting.extract_forecasting_data('tests')
