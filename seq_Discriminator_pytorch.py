#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time    : 17/11/7 下午4:36
# Author  : Shi Bo
# Email   : pkushibo@pku.edu.cn
# File    : seq_Discriminator_pytorch.py

import numpy as np
from sklearn import metrics
import torch


class seqDiscriminator(nn.module):
    """
    Basic class for sequence Discriminator
    """

    def train(self, X, seq_len, Y):
        raise NotImplementedError("Abstract method")

    def test(self, X, seq_len):
        raise NotImplementedError("Abstract method")

    def evaluate(self, X, seq_len, Y):
        raise NotImplementedError("Abstract method")

    def _score(self, Y_true, Y_pred):
        """
        calculate the classification performance of Discriminator on given true labels and predicted probablity
        :param Y_true: ground truth labels
        :param Y_pred: predicted probablity
        :return: a dict of scores
        """
        scoreDict = {}
        scoreDict['AUC'] = metrics.roc_auc_score(Y_true, Y_pred)
        Y_pred[Y_pred >= 0.5] = 1
        Y_pred[Y_pred < 0.5] = 0
        scoreDict['Accuracy'] = metrics.accuracy_score(Y_true, Y_pred)
        confusionMatrix = metrics.confusion_matrix(Y_true, Y_pred)
        scoreDict['Confusion Matrix'] = confusionMatrix
        scoreDict['TPR'] = confusionMatrix[1, 1] / float(confusionMatrix[1, 0] + confusionMatrix[1, 1])
        scoreDict['FPR'] = confusionMatrix[0, 0] / float(confusionMatrix[0, 0] + confusionMatrix[0, 1])
        return scoreDict


class RNN_Classifier(seqDiscriminator):
    """
    RNN_Classifier class to classify given sequences
    """

    def __init__(self, cell_type='LSTM', rnn_layers=[512], ff_layers=[512], attention_layers=[512],
                 is_bidirectional=True, num_tokens=160, max_len=2048, num_class=2, batch_size=128,
                 max_epochs=100, max_epochs_val=5, learning_rate=0.001, model_path='boxD_model'):
        """

        :param cell_type: 'LSTM', 'RNN', 'GRU'
        :param rnn_layers: list of integers indicating layers of rnn
        :param ff_layers: list of integers indicating layers of feed forward network
        :param attention_layers: None or list of integers indicating layers of attention network
        :param is_bidirectional: indicating whether or not to use bidirectional rnn
        :param num_tokens: number of distinct tokens in sequence
        :param max_len: maximum length of a sequence
        :param num_class: number of classification result
        :param batch_size: size of a mini_batch
        :param max_epochs: maximum number of epochs
        :param max_epochs_val: max number of epochs whose eveluation is no better than current best epoch
        :param learning_rate: learning rate for RNN
        :param model_path: the path to save current RNN model
        """
        # set hyper parameters
        self.cell_type = cell_type
        self.rnn_layers = rnn_layers
        self.ff_layers = ff_layers
        self.attention_layers = attention_layers
        self.is_bidirectional = is_bidirectional
        self.num_tokens = num_tokens
        self.max_len = max_len
        self.num_class = num_class
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.max_epochs_val = max_epochs_val
        self.learning_rate = learning_rate
        self.model_path = model_path

        #

































