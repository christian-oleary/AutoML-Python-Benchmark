import logging
import os
import sys

from src.dataset_formatting import DatasetFormatting
from src.forecasting import Forecasting
from src.util import Utils

if __name__ == '__main__': # Needed for any multiprocessing

    forecasting_data_dir = os.path.join('data', 'forecasting')
    anomaly_data_dir = os.path.join('data', 'anomaly_detection')

    # Download data if needed
    gather_metadata = False # If true, will parse all .tsf files and output a .csv file of dataset descriptors
    DatasetFormatting.format_forecasting_data(forecasting_data_dir, gather_metadata=gather_metadata)
    DatasetFormatting.format_anomaly_data(anomaly_data_dir)

    # Run forecasting models
    forecasters = Forecasting.get_forecaster_names()
    Utils.logger.info(f'Available forecasting libraries: {forecasters}')
    Forecasting.run_forecasting_libraries(forecasters, forecasting_data_dir)
