import argparse
from datetime import datetime, timedelta
import os
import time
import warnings

from sklearn.experimental import enable_iterative_imputer # import needed for IterativeImputer

from src.dataset_formatting import DatasetFormatting
from src.forecasting import Forecasting
from src.logs import logger
from tests import gpu_test

if __name__ == '__main__': # Needed for any multiprocessing

    # Start timer
    logger.info(f'Started at {datetime.now().strftime("%d-%m-%y %H:%M:%S")}\n')
    start_time = time.perf_counter()

    # Configuration is set up first
    parser = argparse.ArgumentParser(description='AutoML Python Benchmark')

    parser.add_argument('--anomaly_data_dir', metavar='-A', type=str, nargs='?',
                        default=os.path.join('data', 'anomaly_detection'),
                        help='directory containing anomaly detection datasets')

    parser.add_argument('--global_forecasting_data_dir', metavar='-GF', type=str, nargs='?',
                        default=os.path.join('data', 'global_forecasting'),
                        # default='./tests/data/global/', # test data
                        help='directory containing global forecasting datasets')

    library_options = [
                        'all', # Will run all libraries
                        'installed', # Will run all libraries installed correctly
                        'test', # Will run a test/placholder model

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

    parser.add_argument('--univariate_forecasting_data_dir', metavar='-UF', type=str, nargs='?',
                        default=os.path.join('data', 'univariate_forecasting'),
                        # default='./tests/data/univariate/', # test data
                        help='directory containing univariate forecasting datasets')

    args = parser.parse_args()
    logger.info(f'CLI arguments: {args}')

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

    # Format datasets if needed
    gather_metadata = not os.path.exists(os.path.join(args.global_forecasting_data_dir, '0_metadata.csv'))
    DatasetFormatting.format_global_forecasting_data(args.global_forecasting_data_dir,
                                                        gather_metadata=gather_metadata)
    DatasetFormatting.format_anomaly_data(args.anomaly_data_dir)
    DatasetFormatting.format_univariate_forecasting_data(args.univariate_forecasting_data_dir)

    # Run univariate forecasting models
    data_dir = args.univariate_forecasting_data_dir
    Forecasting().run_forecasting_libraries(data_dir, args, 'univariate')

    # Run global forecasting models
    data_dir = args.global_forecasting_data_dir
    # Forecasting().run_forecasting_libraries(data_dir, args, 'global')

    # Calculate runtime
    logger.info(f'\nFinished at {datetime.now().strftime("%d-%m-%y %H:%M:%S")}')
    duration = timedelta(seconds=time.perf_counter()-start_time)
    logger.debug(f'Total time: {duration}')
