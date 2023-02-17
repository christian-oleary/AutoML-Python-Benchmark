import os
import pytest

from src.utils import Utils


def test_utils_extract_forecasting_data():
    """Test zip file parsing and data extraction"""
    debug=True

    with pytest.raises(NotADirectoryError):
        Utils.extract_forecasting_data('README.md', debug=debug)

    with pytest.raises(IOError):
        Utils.extract_forecasting_data('shell', debug=debug)

    Utils.extract_forecasting_data(os.path.join('data', 'forecasting'), debug=debug)