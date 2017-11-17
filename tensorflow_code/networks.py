#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time    : 17/11/17 下午4:11
# Author  : Shi Bo
# Email   : pkushibo@pku.edu.cn
# File    : networks.py

import numpy as np
import tensorflow as tf

class blackboxDiscriminator():
    """
    black box Discrimanator
    """

    def __init__(self, cell_type='LSTM', rnn_layers=[128], is_bidirectionaal=False,
                 attention_layers=[128], ff_layers=[128], batch_size=128, num_token=161,
                 max_seq_len=2048, num_class=2, scope='blackboxD'):
        """
        initialize black box discriminator
        :param cell_type:
        :param rnn_layers:
        :param is_bidirectionaal:
        :param attention_layers:
        :param ff_layers:
        :param batch_size:
        :param num_token:
        :param max_seq_len:
        :param num_class:
        """
        with tf.variable_scope(scope):
            self.input = tf.placeholder()


