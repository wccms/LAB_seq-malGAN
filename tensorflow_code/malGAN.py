#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time    : 17/11/21 下午4:33
# Author  : Shi Bo
# Email   : pkushibo@pku.edu.cn
# File    : malGAN.py

import numpy as np
import tensorflow as tf


class malGAN():
    """
    class of malGAN, which including a substitute Discriminator and a malware Generator
    """

    def __init__(self, G_cell_type='LSTM', G_rnn_layers=[128], G_is_bidirectional=False, G_attention_layers=None,
                 G_ff_layers=[128], D_cell_type='LSTM', D_rnn_layers=[128], D_is_bidirectional=False,
                 D_attention_layers=[128], D_ff_layers=[128], batch_size=128, num_token=161, max_seq_len=1024,
                 num_class=2, learning_rate=0.001, scope='malGAN', model_path='./black_box_d_model',
                 log_path='./log.txt'):
        """
        initialize malGAN model
        :param G_cell_type:
        :param G_rnn_layers:
        :param G_is_bidirectional:
        :param G_attention_layers:
        :param G_ff_layers:
        :param D_cell_type:
        :param D_rnn_layers:
        :param D_is_bidirectional:
        :param D_attention_layers:
        :param D_ff_layers:
        :param batch_size:
        :param num_token:
        :param max_seq_len:
        :param num_class:
        :param learning_rate:
        :param scope:
        :param model_path:
        :param log_path:
        """

        # define self holding variables
        self.batch_size = batch_size
        self.num_token = num_token
        self.max_seq_len = max_seq_len
        self.num_class = num_class
        self.learning_rate = learning_rate
        self.model_path = model_path
        self.log_path = log_path

        # define network structure
        g_malGAN = tf.Graph()
        with g_malGAN.as_default():
            def generator(scope='')
