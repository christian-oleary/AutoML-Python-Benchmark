"""Call the main function of the module from the command line."""

from __future__ import annotations

import os
import warnings

from loguru import logger
from sklearn.exceptions import ConvergenceWarning
from sklearn.experimental import enable_iterative_imputer  # pylint: disable=W0611  # noqa: F401

from tests import gpu_test
from ml.configuration import Configuration
from ml.tasks import TaskBuilder


def run():
    """Run the main function of the module."""
    warnings.simplefilter('ignore', category=ConvergenceWarning)

    # Load configuration from CLI arguments and environment variables
    config = Configuration()
    logger.info('Starting application')
    config.log_configuration()

    # Check GPU access
    if config.cpu_only:
        logger.info('CPU-only mode: ignoring GPU...')
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Use CPU instead of GPU
    else:
        logger.debug('Checking GPU access...')
        if not gpu_test.tensorflow_test(gpu_required=False):
            logger.warning('TensorFlow cannot access GPU')
        if not gpu_test.pytorch_test(gpu_required=False):
            logger.warning('PyTorch cannot access GPU')

    # Run libraries
    task = TaskBuilder.init_task(config)
    task.run()
    logger.success('Finished execution.')


if __name__ == "__main__":
    run()
