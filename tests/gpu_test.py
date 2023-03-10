import tensorflow as tf
import torch

print('\nTesting TensorFlow...\n')
if tf.test.gpu_device_name():
    print('TensorFlow can access a GPU. Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print('TensorFlow cannot access a GPU')

print('\nTesting PyTorch...\n')
if torch.cuda.is_available() and torch.cuda.device_count() > 0:
    print('PyTorch can access a GPU')
else:
    print('PyTorch cannot access a GPU')
