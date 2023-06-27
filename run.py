from datetime import datetime, timedelta
import os
import time

from src.dataset_formatting import DatasetFormatting
from src.forecasting import Forecasting
from src.util import Utils
from tests import gpu_test

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # Use CPU instead of GPU

if __name__ == '__main__': # Needed for any multiprocessing

    # Ensure GPU access
    assert gpu_test.tensorflow_test(), 'TensorFlow cannot access GPU'
    assert gpu_test.pytorch_test(), 'PyTorch cannot access GPU'

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
    Forecasting.run_forecasting_libraries([
        'AutoGluon', # Python >= 3.8
        'AutoKeras',
        'AutoTS',
        'AutoPyTorch', # Linux
        # 'ETNA', # Not working
        'EvalML',
        'FEDOT',
        'FLAML',
        'Ludwig',
        'PyCaret',
        ],
        forecasting_data_dir, # all datasets
        # './tests/data/forecasting/' # test data
        # './data/other/' # other datasets
        )

    # Calculate runtime
    print(f'\nFinished at {datetime.now().strftime("%d-%m-%y %H:%M:%S")}')
    duration = timedelta(seconds=time.perf_counter()-start_time)
    print(f'Total time: {duration}')
