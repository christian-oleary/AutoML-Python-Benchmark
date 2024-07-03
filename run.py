"""Main application entrypoint"""

import argparse
from datetime import datetime, timedelta
import os
import time
from typing import Any
import warnings

from sklearn.exceptions import ConvergenceWarning
# import needed for IterativeImputer
from sklearn.experimental import enable_iterative_imputer  # pylint: disable=W0611  # noqa

from src.dataset_formatting import DatasetFormatter
from src.forecasting import Forecasting
from src.logs import logger, LogLevel, set_log_dir
from src.validation import Library, Task, Validator


if __name__ == '__main__':  # Needed for any multiprocessing
    warnings.simplefilter('ignore', category=ConvergenceWarning)

    def options_as_str(param_options: list):
        """Parameter options formatting"""
        return ''.join([f'\n- {o}' for o in param_options]) + '\n'

    # Start timer
    start_time = time.perf_counter()

    # Configuration is set up first
    parser = argparse.ArgumentParser(
        description='AutoML Python Benchmark', formatter_class=argparse.RawTextHelpFormatter)

    # CPU Only
    parser.add_argument('--cpu_only', '-CO', action='store_true', help='Only use CPU. No modelling on GPU.\n\n')

    # Data Directory
    parser.add_argument('--data_dir', '-DD', metavar='...', type=str, nargs='?', default=None,
                        help='directory containing datasets\n\n')

    # Libraries
    options = [
        'all',  # Will run all libraries
        'installed',  # Will run all correctly installed libraries
        'test',  # Will run baseline models
        'none',  # No experiments (just other functions)
        *Library.get_options()
    ]
    default: Any = 'installed'
    parser.add_argument('--libraries', '-L', metavar='', type=str.lower, nargs='*',
                        default=default, choices=options,
                        help=f'AutoML libraries to run: {options_as_str(options)}\n\n')

    # Log Level
    options = LogLevel.get_options()
    default = LogLevel.DEBUG.value
    description = f'Log level. (default={default})' + ''.join([f'\n- {o}' for o in options]) + '\n\n'
    parser.add_argument('--log_level', '-LL', metavar='', type=str.lower, nargs='?', default=default,
                        choices=options, help=description)

    # Log Directory
    default = None
    parser.add_argument('--log_dir', '-LD', metavar='...', type=str, nargs='?', default=default,
                        help=f'Directory containing logs (default={default})\n\n')

    # Num. Processes
    default = 1
    parser.add_argument('--nproc', '-N', metavar='...', type=int, nargs='?', default=default,
                        help='Number of processes to allow\n\n')

    # Maximum Results
    default = 1  # i.e. skip if 1 result existing
    parser.add_argument('--max_results', '-MR', metavar='...', type=int, nargs='?', default=default,
                        help='Maximum number of results to generate per library/preset setup\n\n')

    # Repeat Results
    parser.add_argument('--repeat_results', '-RR', action='store_true',
                        help='Train even if some results exist for given experiment\n\n')

    # Results Directory
    default = 'results'
    parser.add_argument('--results_dir', '-RD', metavar='...', type=str, nargs='?', default=default,
                        help='Directory to store results\n\n')

    # Task
    options = Task.get_options()
    default = Task.UNIVARIATE_FORECASTING.value
    parser.add_argument('--task', '-T', metavar='', type=str.lower, nargs='?',
                        default=default, choices=options,
                        help=f'Task type to execute: {options_as_str(options)}\n')

    # Time Limit
    default = 3600  # 1 hour
    parser.add_argument('--time_limit', '-TL', metavar='...', type=int, nargs='?', default=default,
                        help='Time limit in seconds for each library. May not be strictly adhered to.\n\n')

    # Verbosity of Python libraries
    # Scikit-learn: 0 = silent, 3 = maximum information
    # TensorFlow: 0 = silent, 1 = progress bar, 2 = single line
    default = 1
    parser.add_argument('--verbose', '-V', metavar='...', type=int, nargs='?', default=default,
                        help=f'Verbosity of Python libraries (default={default})\n\n')

    ######################################################

    # Parse CLI arguments
    args = parser.parse_args()

    # Validate CLI inputs
    args = Validator().validate_inputs(args)

    # Set log level
    logger.setLevel(args.log_level.upper())
    logger.info(f'Started at {datetime.now().strftime("%d-%m-%y %H:%M:%S")}')

    # Set logging directory (if any)
    if args.log_dir is None:
        logger.warning('No logging dir set. Log directory can be set with --log_dir')
    else:
        set_log_dir()

    # Show CLI argument values
    args_str = '\n-> '.join([f'{arg}: {getattr(args, arg)}' for arg in vars(args)])
    args_str = '\n-> ' + args_str + '\n'
    logger.debug(f'CLI arguments: {args_str}')

    # Check GPU access
    if args.cpu_only:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Use CPU instead of GPU
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
    data_formatter = DatasetFormatter()
    data_formatter.format_data(args)

    # Run libraries
    if None not in args.libraries:
        if Task.is_forecasting_task(args.task):
            Forecasting().run_forecasting_libraries(args)
            Forecasting().analyse_results(args)
        else:
            raise NotImplementedError()

    # Calculate runtime
    logger.info(f'Finished at {datetime.now().strftime("%d-%m-%y %H:%M:%S")}')
    duration = timedelta(seconds=time.perf_counter()-start_time)
    logger.debug(f'Total time: {duration}')
