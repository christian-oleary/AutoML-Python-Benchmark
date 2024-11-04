"""Tests GPU access"""

import os


def tensorflow_test(gpu_required=False) -> bool:
    """Test TensorFlow GPU access"""
    import tensorflow as tf  # pylint: disable=import-outside-toplevel

    print('Testing TensorFlow...')

    device = tf.test.gpu_device_name()
    if device:
        print('TensorFlow can access a GPU.')
        print(f'Default GPU Device: {device}')
    else:
        print('TensorFlow cannot access a GPU')
        if gpu_required:
            raise RuntimeError('TensorFlow cannot access a GPU')
    return device


def pytorch_test(gpu_required=False) -> bool:
    """Test PyTorch GPU access"""
    import torch  # pylint: disable=import-outside-toplevel

    print('Testing PyTorch...')

    access_gpu = torch.cuda.is_available() and torch.cuda.device_count() > 0
    if access_gpu:
        print('PyTorch can access a GPU')
    else:
        print('PyTorch cannot access a GPU')
        if gpu_required:
            raise RuntimeError('PyTorch cannot access a GPU')
    return access_gpu


if __name__ == '__main__':
    tensorflow_test(bool(os.environ.get('TF_GPU_REQUIRED', False)))
    pytorch_test(bool(os.environ.get('TORCH_GPU_REQUIRED', False)))
