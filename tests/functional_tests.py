import os

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
        df = df.drop(['season', 'holiday', 'workingday', 'weather'], axis=1)
        df.to_csv(forecasting_test_path)

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
