"""Main application entrypoint"""

import argparse
from datetime import datetime, timedelta
import os
import time

# import needed for IterativeImputer
from sklearn.experimental import enable_iterative_imputer # pylint: disable=W0611

from src.dataset_formatting import DatasetFormatting
from src.forecasting import Forecasting
from src.logs import logger

if __name__ == '__main__': # Needed for any multiprocessing

    # Start timer
    start_time = time.perf_counter()

    # Configuration is set up first
    parser = argparse.ArgumentParser(description='AutoML Python Benchmark')

    parser.add_argument('--anomaly_data_dir', metavar='-A', type=str, nargs='?',
                        default=os.path.join('data', 'anomaly_detection'),
                        help='directory containing anomaly detection datasets')

    parser.add_argument('--cpu_only', action='store_true', help='Only use CPU. No modelling on GPU.')

    parser.add_argument('--global_forecasting_data_dir', metavar='-GF', type=str, nargs='?',
                        default=os.path.join('data', 'global_forecasting'),
                        # default='./tests/data/global/', # test data
                        help='directory containing global forecasting datasets')

    library_options = [
                        'all', # Will run all libraries
                        'installed', # Will run all libraries installed correctly
                        'test', # Will run baseline models
                        'None', # No experiments (just other functions)

                        'autogluon',
                        'autokeras',
                        'autots',
                        'autopytorch', # Linux/WSL
                        'evalml',
                        'etna',
                        'fedot',
                        'flaml',
                        'ludwig',
                        'pycaret',
                        ]
    parser.add_argument('--libraries', metavar='-L', type=str, nargs='*', default='installed',
                        choices=library_options, help=f'AutoML libraries to run: {library_options}')

    parser.add_argument('--nproc', metavar='-N', type=int, nargs='?', default=1,
                        help='Number of processes to allow')

    parser.add_argument('--repeat_results', action='store_true', help='Train even if results exist for experiment')

    parser.add_argument('--results_dir', metavar='-R', type=str, nargs='?', default='results',
                        help='Directory to store results')

    parser.add_argument('--time_limit', metavar='-T', type=int, nargs='?', default=3600,
                        help='Time limit in seconds for each library. May not be strictly adhered to.')

    parser.add_argument('--univariate_forecasting_data_dir', metavar='-UF', type=str, nargs='?',
                        default=os.path.join('data', 'univariate_libra'),
                        # default=os.path.join('data', 'univariate_electricity'),
                        # default=os.path.join('tests', 'data', 'univariate_electricity'),
                        help='directory containing univariate forecasting datasets')

    args = parser.parse_args()
    logger.info(f'CLI arguments: {args}')

    # Check GPU access
    if args.cpu_only:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # Use CPU instead of GPU
    else:
        logger.info('Checking GPU access...')
        from tests import gpu_test
        if not gpu_test.tensorflow_test():
            logger.warning('TensorFlow cannot access GPU')

        if not gpu_test.pytorch_test():
            logger.warning('PyTorch cannot access GPU')

        # assert gpu_test.tensorflow_test(), 'TensorFlow cannot access GPU'
        # assert gpu_test.pytorch_test(), 'PyTorch cannot access GPU'

    # Format datasets if needed
    DatasetFormatting.format_univariate_forecasting_data(args.univariate_forecasting_data_dir)
    gather_metadata = not os.path.exists(os.path.join(args.global_forecasting_data_dir, '0_metadata.csv'))
    DatasetFormatting.format_global_forecasting_data(args.global_forecasting_data_dir,
                                                        gather_metadata=gather_metadata)
    DatasetFormatting.format_anomaly_data(args.anomaly_data_dir)

    # Run univariate forecasting models
    data_dir = args.univariate_forecasting_data_dir
    if 'None' not in args.libraries:
        Forecasting().run_forecasting_libraries(data_dir, args, 'univariate')
    Forecasting().analyse_results(args, 'univariate')

    # Run global forecasting models
    data_dir = args.global_forecasting_data_dir
    # if 'None' not in args.libraries:
    #     Forecasting().run_forecasting_libraries(data_dir, args, 'global')
    # Forecasting().analyse_results(args, 'global')

    # Calculate runtime
    logger.info(f'Finished at {datetime.now().strftime("%d-%m-%y %H:%M:%S")}')
    duration = timedelta(seconds=time.perf_counter()-start_time)
    logger.debug(f'Total time: {duration}')
