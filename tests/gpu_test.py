"""Tests GPU access"""

import os
import sys


def tensorflow_test(logger, gpu_required=False) -> bool:
    """Test TensorFlow GPU access"""
    import tensorflow as tf  # pylint: disable=import-outside-toplevel
    logger.info('Testing TensorFlow...')

    device = tf.test.gpu_device_name()
    if device:
        logger.info('TensorFlow can access a GPU.')
        logger.info(f'Default GPU Device: {device}')
    else:
        logger.warning('TensorFlow cannot access a GPU')
        if gpu_required:
            raise RuntimeError('TensorFlow cannot access a GPU')
    return device


def pytorch_test(logger, gpu_required=False) -> bool:
    """Test PyTorch GPU access"""
    import torch  # pylint: disable=import-outside-toplevel
    logger.info('Testing PyTorch...')

    access_gpu = torch.cuda.is_available() and torch.cuda.device_count() > 0
    if access_gpu:
        logger.info('PyTorch can access a GPU')
    else:
        logger.warning('PyTorch cannot access a GPU')
        if gpu_required:
            raise RuntimeError('PyTorch cannot access a GPU')
    return access_gpu


if __name__ == '__main__':
    # Add the parent directory to the path
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

    from src.automl.util import logger as logs

    tensorflow_test(logs, bool(os.environ.get('TF_GPU_REQUIRED', False)))
    pytorch_test(logs, bool(os.environ.get('TORCH_GPU_REQUIRED', False)))
