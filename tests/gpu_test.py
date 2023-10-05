"""Tests GPU access"""

import tensorflow as tf
import torch

def tensorflow_test(debug=False):
    """Test TensroFlow GPU access"""
    if debug:
        print('\nTesting TensorFlow...\n')

    if tf.test.gpu_device_name():
        if debug:
            print(f'TensorFlow can access a GPU. Default GPU Device: {tf.test.gpu_device_name()}')
        access_gpu = True
    else:
        if debug:
            print('TensorFlow cannot access a GPU')
        access_gpu = False
    return access_gpu

def pytorch_test(debug=False):
    """Test PyTorch GPU access"""
    if debug:
        print('\nTesting PyTorch...\n')

    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        if debug:
            print('PyTorch can access a GPU')
        access_gpu = True
    else:
        if debug:
            print('PyTorch cannot access a GPU')
        access_gpu = False
    return access_gpu

if __name__ == '__main__':
    tensorflow_test(True)
    pytorch_test(True)
