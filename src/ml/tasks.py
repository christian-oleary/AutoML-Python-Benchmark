"""Module defining abstract and concrete machine learning tasks."""

from abc import ABC, abstractmethod

from loguru import logger

from ml import TaskName
from ml.configuration import Configuration
from ml.dataset import DataFormatter, DatasetReader


class AbstractTask(ABC):
    """Abstract base class for different machine learning tasks."""

    def __init__(self, config: Configuration):
        self.config = config

    def read_data(self):
        """Read the data for the task."""
        self.dataset = DatasetReader(self.config).init_dataset()
        logger.debug('Dataset ready')

    def prepare_data(self):
        """Prepare/preprocess the data for the task."""
        self.dataset = DataFormatter(self.config).preprocess(self.dataset)

    @abstractmethod
    def call_libraries(self):
        """Call the necessary AutoML libraries for the task."""

    @abstractmethod
    def evaluate_results(self):
        """Evaluate the results of the task."""

    def run(self):
        """Run the task."""
        self.read_data()
        self.prepare_data()
        self.call_libraries()
        self.evaluate_results()


class PrepareDataTask(AbstractTask):
    """Task to download and preprocess data without modelling."""

    def call_libraries(self):
        """No modelling."""

    def evaluate_results(self):
        """No modelling."""


class TSClassificationTask(AbstractTask):
    """Time series classification task."""

    def call_libraries(self):
        """Call the necessary AutoML libraries for the task."""
        if len(self.config.libraries) == 0:
            logger.warning('No libraries specified. Skipping training...')

    def evaluate_results(self):
        """Evaluate the results of the task."""
        if len(self.config.libraries) == 0:
            logger.warning('No libraries specified. Skipping evaluation...')


class TaskBuilder:
    """Factory class to create task instances based on the task name."""

    @staticmethod
    def init_task(config: Configuration) -> AbstractTask:
        """Get the task instance based on the task name.

        :param Configuration config: Configuration object.
        :return AbstractTask: Instance of the task.
        """
        logger.info(f'Initializing time series task: "{TaskName.CLASSIFICATION}"')
        if config.task == TaskName.CLASSIFICATION:
            return TSClassificationTask(config)
        if config.task == TaskName.PREPARE_DATA:
            return PrepareDataTask(config)
        raise ValueError(f'Unknown task name: "{config.task}"')
