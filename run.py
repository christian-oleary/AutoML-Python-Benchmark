import argparse
from datetime import datetime, timedelta
import os
import time
import warnings

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

    parser.add_argument('--global_forecasting_data_dir', metavar='-GF', type=str, nargs='?',
                        default=os.path.join('data', 'global_forecasting'),
                        # default='./tests/data/forecasting/', # test data
                        help='directory containing global forecasting datasets')

    library_options = [
                        'all', # Will run all libraries
                        'installed', # Will run all libraries installed correctly

                        'autogluon',
                        'autokeras',
                        'autots',
                        'autopytorch', # Linux/WSL
                        'evalml',
                        # 'etna', # Internal Library Errors
                        'fedot',
                        'flaml',
                        'ludwig',
                        'pycaret',
                        ]
    parser.add_argument('--libraries', metavar='-L', type=str, nargs='*', default='installed',
                        choices=library_options, help=f'AutoML libraries to run: {library_options}')

    parser.add_argument('--nproc', metavar='-N', type=int, nargs='?', default=1,
                        help='Number of CPU processes to allow')

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
    print('CLI arguments:', args)

    # Check GPU access
    if args.use_gpu:
        if not gpu_test.tensorflow_test():
            warnings.warn('TensorFlow cannot access GPU')

        if not gpu_test.pytorch_test():
            warnings.warn('PyTorch cannot access GPU')

        # assert gpu_test.tensorflow_test(), 'TensorFlow cannot access GPU'
        # assert gpu_test.pytorch_test(), 'PyTorch cannot access GPU'
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # Use CPU instead of GPU

    # Download data if needed
    exp_dir = os.path.join('data', 'global_forecasting')
    if args.global_forecasting_data_dir == exp_dir:
        gather_metadata = not os.path.exists(os.path.join(exp_dir, '0_metadata.csv'))
        DatasetFormatting.format_forecasting_data(args.global_forecasting_data_dir, gather_metadata=gather_metadata)

    exp_dir = os.path.join('data', 'anomaly_detection')
    if args.anomaly_data_dir == exp_dir:
        gather_metadata = not os.path.exists(os.path.join(exp_dir, '0_metadata.csv'))
        DatasetFormatting.format_anomaly_data(args.anomaly_data_dir)

    # Run forecasting models
    Forecasting.run_global_forecasting_libraries(args)

    # Calculate runtime
    print(f'\nFinished at {datetime.now().strftime("%d-%m-%y %H:%M:%S")}')
    duration = timedelta(seconds=time.perf_counter()-start_time)
    print(f'Total time: {duration}')
