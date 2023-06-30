import argparse
from datetime import datetime, timedelta
import os
import time

from src.dataset_formatting import DatasetFormatting
from src.forecasting import Forecasting
from tests import gpu_test

if __name__ == '__main__': # Needed for any multiprocessing

    # Start timer
    print(f'Started at {datetime.now().strftime("%d-%m-%y %H:%M:%S")}\n')
    start_time = time.perf_counter()

    # Configuration is set up first
    parser = argparse.ArgumentParser(description='AutoML Python Benchmark')

    parser.add_argument('--anomaly_data_dir', metavar='-A', type=str, nargs='?',
                        default=os.path.join('data', 'anomaly_detection'),
                        help='directory containing anomaly detection datasets')

    parser.add_argument('--forecasting_data_dir', metavar='-F', type=str, nargs='?',
                        default=os.path.join('data', 'forecasting'),
                        # default='./tests/data/forecasting/', # test data
                        # default='./data/other/', # other datasets
                        help='directory containing forecasting datasets')

    parser.add_argument('--libraries', metavar='-L', type=str, nargs='*', default='installed',
                        choices=[
                            'all', # Will run all libraries
                            'installed', # Will run all libraries installed correctly

                            'AutoGluon', # Python >= 3.8
                            'AutoKeras',
                            'AutoTS',
                            'AutoPyTorch', # Linux
                            'EvalML',
                            # 'ETNA', # Not working
                            'FEDOT',
                            'FLAML',
                            'Ludwig',
                            'PyCaret',
                            ],
                        help='AutoML libraries to run')

    parser.add_argument('--n_cores', metavar='-N', type=int, nargs='?', default=1,
                        help='Number of CPU cores to allow')

    parser.add_argument('--preset', metavar='-P', type=str, nargs='?', default='best',
                        choices=['best', 'fastest'],
                        help='Library setup to use: "best" or "fastest". May be ignored.')

    parser.add_argument('--results_dir', metavar='-R', type=str, nargs='?', default='results',
                        help='Directory to store results')

    parser.add_argument('--time_limit', metavar='-T', type=int, nargs='?', default=3600,
                        help='Time limit in seconds for each library. May not be strictly adhered to.')

    parser.add_argument('--use_gpu', metavar='-G', type=bool, nargs='?', default=True,
                        help='Boolean to decide if libraries can use GPU')

    args = parser.parse_args()
    print(args)

    # GPU access
    if args.use_gpu:
        assert gpu_test.tensorflow_test(), 'TensorFlow cannot access GPU'
        assert gpu_test.pytorch_test(), 'PyTorch cannot access GPU'
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # Use CPU instead of GPU

    assert gpu_test.tensorflow_test(), 'TensorFlow cannot access GPU'
    assert gpu_test.pytorch_test(), 'PyTorch cannot access GPU'

    # Download data if needed
    gather_metadata = not os.path.exists('./data/forecasting/0_metadata.csv')
    DatasetFormatting.format_forecasting_data(args.forecasting_data_dir, gather_metadata=gather_metadata)
    DatasetFormatting.format_anomaly_data(args.anomaly_data_dir)

    # Run forecasting models
    Forecasting.run_forecasting_libraries(args)

    # Calculate runtime
    print(f'\nFinished at {datetime.now().strftime("%d-%m-%y %H:%M:%S")}')
    duration = timedelta(seconds=time.perf_counter()-start_time)
    print(f'Total time: {duration}')
