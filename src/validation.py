"""File for validating program inputs"""

import os

from src.logs import BaseEnum, logger


class Task(BaseEnum):
    """Machine Learning Tasks"""

    UNIVARIATE_FORECASTING = 'univariate'
    MULTIVARIATE_FORECASTING = 'multivariate'
    GLOBAL_FORECASTING = 'global'
    ANOMALY_DETECTION = 'anomaly_detection'

    @classmethod
    def is_forecasting_task(cls, value):
        """Determine if task involves time series forecasting"""
        forecasting_tasks = [
            cls.UNIVARIATE_FORECASTING.value,
            cls.MULTIVARIATE_FORECASTING.value,
            cls.GLOBAL_FORECASTING.value,
        ]
        # print('value.__str__()', value.__str__(), type(value.__str__()))
        # print('forecasting_tasks', forecasting_tasks)
        # print( value.__str__().lower() in forecasting_tasks)
        # return value.__str__().lower() in forecasting_tasks
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
        if args.task in [ 'multivariate', 'global' ]:
            raise NotImplementedError('multivariate forecasting not implemented')

        # Libraries
        if args.libraries == 'all':
            args.libraries = Task.get_options()

        elif args.libraries == 'installed':
            args = self.check_installed(args)

        else:
            if not isinstance(args.libraries, list):
                raise TypeError(
                    f'libraries must be a list or string. Received: {type(args.libraries)}')

            for library in args.libraries:
                if library not in [ 'none', 'test' ] and library not in Library.get_options():
                    raise ValueError(f'Unknown library. Options: {Library.get_options()}')

        if not isinstance(args.nproc, int):
            raise TypeError(f'nproc must be an int. Received {type(args.nproc)}')

        if args.nproc == 0 or args.nproc < -1:
            raise ValueError(f'nproc must be -1 or a positive int. Received {args.nproc}')

        if not isinstance(args.results_dir, str) and args.results_dir != None:
            raise TypeError(
                f'results_dir must be str/None. Received: {args.results_dir} ({type(args.results_dir)})')

        if args.data_dir is None:
            raise TypeError('data_dir must not be None')

        try:
            os.listdir(args.data_dir)
        except NotADirectoryError as e:
            raise NotADirectoryError(
                f'Unknown directory for data_dir. Received: {args.data_dir}') from e

        if not isinstance(args.time_limit, int):
            raise TypeError(f'time_limit must be an int. Received: {args.time_limit}')

        if args.time_limit <= 0:
            raise ValueError(f'time_limit must be > 0. Received: {args.time_limit}')

        if args.verbose <= 0:
            raise ValueError(f'verbose must be > 0. Received: {args.time_limit}')

        return args


    def check_installed(self, args):
        """Determine which libararies are installed.

        :param args: Argparser configuration
        :raises ValueError: If no AutoML library is installed
        :return: Updated argparser args object
        """
        # Try to import libraries to determine which are installed
        args.libraries = []
        try:
            from src.autogluon.models import AutoGluonForecaster # noqa # pylint: disable=unused-import
            args.libraries.append('autogluon')
            logger.debug('Imported AutoGluon')
        except ModuleNotFoundError as e:
            logger.debug('Not using AutoGluon')

        try:
            from src.autokeras.models import AutoKerasForecaster # noqa # pylint: disable=unused-import
            args.libraries.append('autokeras')
            logger.debug('Imported AutoKeras')
        except ModuleNotFoundError as e:
            logger.debug('Not using AutoKeras')

        try:
            from src.autots.models import AutoTSForecaster # noqa # pylint: disable=unused-import
            args.libraries.append('autots')
            logger.debug('Imported AutoTS')
        except ModuleNotFoundError as e:
            logger.debug('Not using AutoTS')

        try:
            from src.autopytorch.models import AutoPyTorchForecaster # noqa # pylint: disable=unused-import
            args.libraries.append('autopytorch')
            logger.debug('Imported AutoPyTorch')
        except ModuleNotFoundError as e:
            logger.debug('Not using AutoPyTorch')

        try:
            from src.evalml.models import EvalMLForecaster # noqa # pylint: disable=unused-import
            args.libraries.append('evalml')
            logger.debug('Imported EvalML')
        except ModuleNotFoundError as e:
            logger.debug('Not using EvalML')

        try:
            from src.etna.models import ETNAForecaster # noqa # pylint: disable=unused-import
            args.libraries.append('etna')
            logger.debug('Imported ETNA')
        except ModuleNotFoundError as e:
            logger.debug('Not using ETNA')

        try:
            from src.fedot.models import FEDOTForecaster # noqa # pylint: disable=unused-import
            args.libraries.append('fedot')
            logger.debug('Imported FEDOT')
        except ModuleNotFoundError as e:
            logger.debug('Not using FEDOT')

        try:
            from src.flaml.models import FLAMLForecaster # noqa # pylint: disable=unused-import
            args.libraries.append('flaml')
            logger.debug('Imported FLAML')
        except:
            logger.debug('Not using FLAML')

        try:
            from src.ludwig.models import LudwigForecaster # noqa # pylint: disable=unused-import
            args.libraries.append('ludwig')
            logger.debug('Imported Ludwig')
        except ModuleNotFoundError as e:
            logger.debug('Not using Ludwig')

        try:
            from src.pycaret.models import PyCaretForecaster # noqa # pylint: disable=unused-import
            args.libraries.append('pycaret')
            logger.debug('Imported PyCaret')
        except ModuleNotFoundError as e:
            logger.debug('Not using PyCaret')

        if len(args.libraries) == 0:
            raise ModuleNotFoundError('No AutoML libraries can be imported. Are any installed?')

        logger.info(f'Using Libraries: {args.libraries}')
        return args
