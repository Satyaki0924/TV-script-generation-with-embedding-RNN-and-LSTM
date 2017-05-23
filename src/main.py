import warnings
from distutils.version import LooseVersion

import tensorflow as tf

from src.test import Test
from src.train import Train


class Main():
    @staticmethod
    def assert_v(num_epochs, batch_size, rnn_size, embed_dim, seq_length, learning_rate, show_every_n_batches, ip):
        assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer'
        print('TensorFlow Version: {}'.format(tf.__version__))
        if not tf.test.gpu_device_name():
            warnings.warn('No GPU found. Please use a GPU to train your neural network.')
            print('*** THIS MODEL REQUIRES A GPU TO RUN. YOU ARE USING CPU. ***')
        else:
            print('*****Author: Satyaki Sanyal*****')
            print('***This project must only be used for educational purpose***')
            if ip == 1:
                Train(num_epochs, batch_size, rnn_size, embed_dim, seq_length, learning_rate,
                      show_every_n_batches).train()
            elif ip == 2:
                Test().test()
            print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
