from datetime import datetime, timedelta
import os
import time

from src.dataset_formatting import DatasetFormatting
from src.forecasting import Forecasting
from src.util import Utils

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # Use CPU instead of GPU

if __name__ == '__main__': # Needed for any multiprocessing

    # Start timer
    print(f'Started at {datetime.now().strftime("%d-%m-%y %H:%M:%S")}\n')
    start_time = time.perf_counter()

    # Data directories
    forecasting_data_dir = os.path.join('data', 'forecasting')
    anomaly_data_dir = os.path.join('data', 'anomaly_detection')

    # Download data if needed
    gather_metadata = not os.path.exists('./data/forecasting/0_metadata.csv')
    DatasetFormatting.format_forecasting_data(forecasting_data_dir, gather_metadata=gather_metadata)
    DatasetFormatting.format_anomaly_data(anomaly_data_dir)

    # Run forecasting models
    forecasters = Forecasting.get_forecaster_names()
    Utils.logger.info(f'Available forecasting libraries: {forecasters}')
    # Forecasting.run_forecasting_libraries(forecasters, './tests/data/forecasting/')
    Forecasting.run_forecasting_libraries([
        'AutoGluon', # Python >= 3.8
        'AutoKeras',
        'AutoTS',
        'AutoPyTorch', # Linux
        'EvalML', # evalml > 0.43
        'ETNA', # Not working
        'FEDOT',
        'FLAML',
        'Ludwig',
        'PyCaret',
        ],
        forecasting_data_dir, # all datasets
        # './tests/data/forecasting/' # test data
        )

    # Calculate runtime
    print(f'\nFinished at {datetime.now().strftime("%d-%m-%y %H:%M:%S")}')
    duration = timedelta(seconds=time.perf_counter()-start_time)
    print(f'Total time: {duration}')
