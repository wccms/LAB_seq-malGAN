#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time    : 17/11/17 下午4:11
# Author  : Shi Bo
# Email   : pkushibo@pku.edu.cn
# File    : networks.py

from datetime import datetime
import numpy as np
import tensorflow as tf

from utils import write_log


class blackboxDiscriminator():
    """
    black box Discrimanator
    """

    def __init__(self, cell_type='LSTM', rnn_layers=[128], is_bidirectionaal=False,
                 attention_layers=[128], ff_layers=[128], batch_size=128, num_token=161,
                 max_seq_len=2048, num_class=2, learning_rate=0.001, scope='black_box_D'):
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
            self.input = tf.placeholder(tf.int32, [batch_size, max_seq_len])
            self.input_len = tf.placeholder(tf.int32, [batch_size])
            self.target = tf.placeholder(tf.int32, [batch_size])

            # decide cell type
            if cell_type == 'LSTM':
                cell = tf.contrib.rnn.BasicLSTMCell
            elif cell_type == 'RNN':
                cell = tf.contrib.rnn.BasicRNNCell
            elif cell_type == 'GRU':
                cell = tf.contrib.rnn.GRUCell
            else:
                raise ValueError('cell_type must be in ["LSTM", "RNN", "GRU"]')

            # build network structure: rnn part
            self.input_onehot = tf.one_hot(self.input, num_token)  # batch_size * max_seq_len * num_token
            if len(rnn_layers) == 1:
                rnn_cell = cell(rnn_layers[0])
            else:
                rnn_cell = [cell(layer) for layer in rnn_layers]
                rnn_cell = tf.contrib.rnn.MultiRNNCell(rnn_cell)
            if is_bidirectionaal:
                (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(rnn_cell, rnn_cell, self.input_onehot,
                                                                            self.input_len, dtype=tf.float32)
                if attention_layers is None:
                    output_fw = tf.reverse_sequence(output_fw, self.input_len, 1, 0)
                    output = tf.concat([output_fw[:, 0, :], output_bw[:, 0, :]], 1)  # batch_size * hidden_dim
                else:
                    output = tf.concat([output_fw, output_bw], 2)  # batch_size * max_seq_len * hidden_dim
            else:
                output = tf.nn.dynamic_rnn(rnn_cell, self.input_onehot, self.input_len, dtype=tf.float32)
                if attention_layers is None:
                    output = output[:, 0, :]  # batch_size * hidden_dim
            # build network structure: attention part
            if attention_layers is not None:
                attention_layers = attention_layers + [1]
                for i, layer in enumerate(attention_layers):
                    if i < len(attention_layers) - 1:
                        attention_weight = tf.contrib.layers.fully_connected(output, layer, activation_fn=tf.nn.tanh)
                    else:
                        attention_weight = tf.contrib.layers.fully_connected(output, layer, activation_fn=None)
                attention_weight = tf.exp(tf.squeeze(attention_weight))
                input_mask = tf.sequence_mask(self.input_len, max_seq_len)
                attention_weight = attention_weight * input_mask
                attention_weight_sum = tf.reduce_sum(attention_weight, 1, keep_dims=True)
                attention_weight /= attention_weight_sum
                output = tf.reduce_sum(output * tf.expand_dims(attention_weight, 2), axis=1)
            # build network structure: feed forward part
            # output size = batch_size * hidden_dim
            ff_layers += [num_class]
            for i, layer in ff_layers:
                if i < len(attention_layers) - 1:
                    output = tf.contrib.layers.fully_connected(output, layer, activation_fn=tf.nn.tanh)
                else:
                    output = tf.contrib.layers.fully_connected(output, layer, activation_fn=None)
            self.output = tf.nn.softmax(output)  # size = batch_size * num_class

            # calculate loss and define optimizer
            self.loss = tf.reduce_sum(
                tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.target, logits=self.output))
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            grads_and_vars = optimizer.compute_gradients(self.loss)
            grads_and_vars = [(tf.clip_by_value(grad, -0.1, 0.1), var) for (grad, var) in grads_and_vars]
            self.train_opt = optimizer.apply_gradients(grads_and_vars)

    def train(self, X, seq_len, Y, batch_size=128, max_epochs=1000):
        """
        train model
        :param X:
        :param seq_len:
        :param Y:
        :param batch_size:
        :param max_epochs:
        :return:
        """
        # shuffle data
        indexes = np.random.shuffle(np.arange(len(X)))
        X = X[indexes], seq_len = seq_len[indexes], Y = Y[indexes]
        num_train = len(X) * 0.8
        X_val = X_val[num_train:], seq_len_val = seq_len[num_train:], Y_val = Y[num_train:]
        X = X[:num_train], seq_len = seq_len[:num_train], Y = Y[:num_train]

        # training for max_epochs
        for epoch_i in range(max_epochs):
            pass
