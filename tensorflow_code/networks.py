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
                 max_seq_len=2048, num_class=2, learning_rate=0.001, scope='black_box_D',
                 model_path='./black_box_d_model'):
        """

        :param cell_type:
        :param rnn_layers:
        :param is_bidirectionaal:
        :param attention_layers:
        :param ff_layers:
        :param batch_size:
        :param num_token:
        :param max_seq_len:
        :param num_class:
        :param learning_rate:
        :param scope:
        :param model_path:
        """
        self.batch_size = batch_size
        self.num_token=num_token
        self.max_seq_len=max_seq_len
        self.learning_rate=learning_rate
        self.num_class=num_class
        self.model_path = model_path
        # define network structure
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
            self.init_opt = tf.global_variables_initializer()

    def train(self, X, seq_len, Y, batch_size=128, max_epochs=100, max_epochs_val=5):
        """
        train model
        :param X:
        :param seq_len:
        :param Y:
        :param batch_size:
        :param max_epochs:
        :return:
        """

        # get session and saver
        self.sess = tf.Session()
        self.saver = tf.train.Saver()

        # shuffle data
        indexes = np.arange(len(X))
        np.random.shuffle(indexes)
        X = X[indexes], seq_len = seq_len[indexes], Y = Y[indexes]
        num_train = len(X) * 0.8
        X_val = X_val[num_train:], seq_len_val = seq_len[num_train:], Y_val = Y[num_train:]
        X = X[:num_train], seq_len = seq_len[:num_train], Y = Y[:num_train]

        # training for max_epochs
        best_val_loss = 9999.0
        best_val_epoch = 0
        self.sess.run(self.init_opt)
        for epoch_i in range(max_epochs):
            train_loss = 0.0
            for start, end in zip(range(0, len(X), batch_size), range(batch_size, len(X) + 1, batch_size)):
                X_batch = X[start:end], seq_len_batch = seq_len[start:end], Y_batch = Y[start:end]
                _, loss = self.sess.run([self.train_opt, self.loss],
                                        feed_dict={self.input: X_batch, self.input_len: seq_len_batch,
                                                   self.target: Y_batch})
                train_loss += loss
            val_loss = 0.0
            for start, end in zip(0, len(X_val), batch_size), range(batch_size, len(X_val) + 1, batch_size):
                X_val_batch = X_val[start:end]
                seq_len_val_batch = seq_len_val[start:end]
                Y_val_batch = Y_val[start:end]
                _, loss = self.sess.run(self.loss,
                                        feed_dict={self.input: X_val_batch, self.input_len: seq_len_val_batch,
                                                   self.target: Y_val_batch})
                val_loss += loss
            print('training black box D: epoch=%d\ttrain_loss=%g\tval_loss=%g' % (epoch_i, train_loss, val_loss))
            self.saver.save(self.sess, self.model_path, epoch_i)
            if val_loss < best_val_epoch:
                best_val_epoch = epoch_i
                best_val_loss = val_loss
            if epoch_i - best_val_loss > max_epochs_val:
                self.saver.restore(self.sess, self.model_path + '-' + str(best_val_epoch))

    def predict_proba(self, X, seq_len):
        """
        predict probablity for given X and seq_len
        :param X:
        :param seq_len:
        :return:
        """
        proba = np.zeros((len(X),self.num_class))
        last_end=0
        for start, end in zip(range(0, len(X), self.batch_size), range(self.batch_size, len(X) + 1, self.batch_size)):
            X_batch = X[start:end], seq_len_batch = seq_len[start:end]
            proba_batch = self.sess.run(self.output,feed_dict={self.input:X_batch,self.input_len:seq_len_batch})
            proba[start:end]=proba_batch
            last_end=end
        X_batch = X[last_end:], seq_len_batch = seq_len[last_end]
        proba_batch = self.sess.run(self.output, feed_dict={self.input: X_batch, self.input_len: seq_len_batch})
        proba[last_end:] = proba_batch
        return proba