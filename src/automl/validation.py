"""File for validating program inputs"""

import importlib
import os

from src.logs import BaseEnum, logger


class Task(BaseEnum):
    """Machine Learning Tasks"""

    ANOMALY_DETECTION = 'anomaly_detection'
    CLASSIFICATION = 'classification'
    GLOBAL_FORECASTING = 'global'
    MULTIVARIATE_FORECASTING = 'multivariate'
    NONE = 'none'
    UNIVARIATE_FORECASTING = 'univariate'

    @classmethod
    def is_forecasting_task(cls, value):
        """Determine if task involves time series forecasting"""
        forecasting_tasks = [
            cls.UNIVARIATE_FORECASTING.value,
            cls.MULTIVARIATE_FORECASTING.value,
            cls.GLOBAL_FORECASTING.value,
        ]
        return value.lower() in forecasting_tasks


class Library(BaseEnum):
    """AutoML Libraries"""

    AutoGluon = 'autogluon'
    AutoKeras = 'autokeras'
    AutoTS = 'autots'
    AutoPyTorch = 'autopytorch'
    ETNA = 'etna'
    EvalML = 'evalml'
    FEDOT = 'fedot'
    FLAML = 'flaml'
    Ludwig = 'ludwig'
    PyCaret = 'pycaret'


class Validator:
    """Class for validating parameters"""

    def validate_inputs(self, args):
        """Validate CLI arguments

        :param argparse.Namespace args: arguments from command line
        """

        # Task
        if args.task in ['multivariate', 'global']:
            raise NotImplementedError('multivariate forecasting not implemented')

        # Libraries
        if args.libraries == 'all':
            args.libraries = Task.get_options()

        elif args.libraries == 'installed':
            args = self.check_installed(args)

        else:
            if not isinstance(args.libraries, list):
                raise TypeError(
                    f'libraries must be a list or string. Received: {type(args.libraries)}'
                )

            for library in args.libraries:
                if library not in ['none', 'test'] and library not in Library.get_options():
                    raise ValueError(f'Unknown library. Options: {Library.get_options()}')

        if not isinstance(args.nproc, int):
            raise TypeError(f'nproc must be an int. Received {type(args.nproc)}')

        if args.nproc == 0 or args.nproc < -1:
            raise ValueError(f'nproc must be -1 or a positive int. Received {args.nproc}')

        if not isinstance(args.results_dir, str) and args.results_dir is not None:
            raise TypeError(
                f'results_dir must be str/None. Received: {args.results_dir} ({type(args.results_dir)})'
            )

        if args.data_dir is None:
            raise TypeError('data_dir must not be None')

        try:
            os.listdir(args.data_dir)
        except NotADirectoryError as e:
            raise NotADirectoryError(
                f'Unknown directory for data_dir. Received: {args.data_dir}'
            ) from e

        if not isinstance(args.time_limit, int):
            raise TypeError(f'time_limit must be an int. Received: {args.time_limit}')

        if args.time_limit <= 0:
            raise ValueError(f'time_limit must be > 0. Received: {args.time_limit}')

        if args.verbose <= 0:
            raise ValueError(f'verbose must be > 0. Received: {args.time_limit}')

        return args

    def check_installed(self, args):
        """Determine which libraries are installed.

        :param args: argparse configuration
        :raises ValueError: If no AutoML library is installed
        :return: Updated argparse args object
        """
        # Try to import libraries to determine which are installed
        args.libraries = []
        for library in [lib.value for lib in Library]:
            try:
                importlib.import_module(f'src.{library}.models', __name__)
                args.libraries.append(library)
                logger.debug(f'Imported {library}')
            except ModuleNotFoundError:
                logger.debug(f'Not using {library}')

        if len(args.libraries) == 0:
            raise ModuleNotFoundError('No AutoML libraries can be imported. Are any installed?')

        logger.info(f'Using Libraries: {args.libraries}')
        return args
