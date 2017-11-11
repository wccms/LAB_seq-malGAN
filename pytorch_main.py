#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time    : 2017/11/9 21:42
# Author  : Shi Bo
# File    : pytorch_main.py

import numpy as np
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from datetime import datetime
import sys
import shutil
from pytorch_Networks import seqDiscriminator
from utils import load_dataset
from utils import log_writer
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable


def train_D(D, X, seqLen, Y, batch_size=128, max_epochs=1000, max_epochs_val=5, learning_rate=0.001, log_path):
    """
    train discriminator D given X, seqLen and Y
    :param D: Discriminator model
    :param X: data X with shape of [batch_size * seq_len], vablus in X are within range [0,vocab_size=161]
    :param seqLen: sequence length with shape of [batch_size]
    :param Y: data Y with shape of [batch_size]
    :param batch_size: batch size
    :param max_epochs: max epoch number
    :param max_epochs_val: epoch number when model do not improve any more
    :param learning_rate: learning rate
    :return: ???
    """

    # split X, seqLen and Y into training_set adn validation set
    train_set_ratio = 0.75
    shuffled_index = np.random.shuffle(np.arange(len(X)))
    X = X[shuffled_index]
    seqLen = seqLen[shuffled_index]
    Y = Y[shuffled_index]
    train_set_len = len(X) * train_set_ratio
    X_val = X[train_set_len:]
    seqLen_val = seqLen[train_set_len:]
    Y_val = Y[train_set_len:]
    X = X[:train_set_len]
    seqLen = seqLen[:train_set_len]
    Y = Y[:train_set_len]

    # start traing epochs
    lossF = nn.CrossEntropyLoss()
    optimizer = optim.Adam(lr=learning_rate)
    best_val_loss = 1000.0
    best_val_epoch = 0
    for epoch_i in range(max_epochs):
        train_loss = 0.0
        for start, end in zip(range(0, len(X), batch_size), range(batch_size, len(X) + 1, batch_size)):
            batch_X = Variable(torch.from_numpy(X[start:end]))
            batch_seqLen = Variable(torch.from_numpy(seqLen[start:end]))
            batch_Y = Variable(torch.from_numpy(Y[start:end]))
            D.zero_grad()
            h0, c0 = D.init_hidden()
            batch_out = D(batch_X, batch_seqLen, h0, c0)
            loss = lossF(batch_out, batch_Y)
            train_loss += loss
            loss.back_ward()
            optimizer.step()
        train_loss /= len(X)
        val_loss = 0.0
        for start, end in zip(range(0, len(X_val), batch_size), range(batch_size, len(X_val) + 1, batch_size)):
            batch_X = Variable(torch.from_numpy(X_val[start:end]))
            batch_seqLen = Variable(torch.from_numpy(seqLen_val[start:end]))
            batch_Y = Variable(torch.from_numpy(Y_val[start:end]))
            D.zero_grad()
            h0, c0 = D.init_hidden()
            batch_out = D(batch_X, batch_seqLen, h0, c0)
            loss = lossF(batch_out, batch_Y)
            val_loss += loss
            loss.back_ward()
            optimizer.step()
        val_loss /= len(X_val)

        # judge current validation loss, if no better than best val loss for max_epoch_val, break
        # replace best_val_loss if current val_loss is less than best_val_loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_epoch = epoch_i
        log_message = 'traing Discriminator: %d epoch, train_loss = %g, val_loss = %g' \
                      % (epoch_i, train_loss, val_loss)
        log_writer(log_path, log_message)

        if epoch_i - best_val_epoch >= max_epochs_val:
            break


def train_seq_malGAN():
    """
    main training function: first train subD, then alternately train boxD and malG
    :return: None
    """
    max_seq_len = 1024

    # make workspace directory for current mission and copy code
    timeTag = datetime.now().strftime('%Y-%m-%d_%H:%M')
    dir_path = '../pytorch_result/'
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    dir_path = '../pytorch_result/' + timeTag
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    if os.path.exists(os.path.join(dir_path, 'code')):
        shutil.rmtree(os.path.join(dir_path, 'code'))
    shutil.copytree('.', os.path.join(dir_path, 'code'))
    log_filepath = dir_path + 'log.txt'
    score_template = 'TPR %(TPR)f\tFPR %(FPR)f\tAccuracy %(Accuracy)f\tAUC %(AUC)f'
    log_message = str(datetime.now()) + 'Start training seq_malGAN.\n'

    # load data
    X_malware, seqLen_malware, X_benigh, seqLen_benigh = \
        load_dataset('../data/API_rand_trainval_len_2048.txt', max_seq_len, 0)
    X = np.vstack((X_malware, X_benigh))
    seqLen = np.hstack((seqLen_malware, seqLen_benigh))
    Y = np.array([1] * len(X_malware) + [0] * len(X_benigh))
    X_malware_test, seqLen_malware_test, X_benigh_test, seqLen_benigh_test = \
        load_dataset('../data/API_rand_test_len_2048.txt', max_seq_len, 0)
    X_test = np.vstack((X_malware_test, X_malware_test))
    seqLen_test = np.hstack((seqLen_malware_test, seqLen_benigh_test))
    Y_test = np.array([1] * len(X_malware_test) + [0] * len(X_benigh_test))
    log_message += str(datetime.now()) + '\tFinish loading data.\n'

    # define substituteD as subD, black box D as boxD and malware Genarator as G
    subD_params = {'vocab_num': 160, 'embedding_dim': 160, 'hidden_dim': 128, 'is_bidirectional': False,
                   'max_seq_len': 1024, 'attention_layers': None, 'ff_layers': [512], 'class_num': 2}
    subD = seqDiscriminator(**subD_params)
    # boxD_params = {'vocab_num': 160, 'embedding_dim': 160, 'hidden_dim': 128, 'is_bidirectional': False,
    #                'max_seq_len': 1024, 'attention_layers': None, 'ff_layers': [512], 'class_num': 2}
    # G_params = {}
    log_message += str(datetime.now()) + '\tFinish defining subD, boxD and G.\n'
    log_writer(log_filepath, log_message)

    # train substitute Discrimanator first
    log_message = str(datetime.now()) + '\n1. training sequence Discriminator.\n'
    train_D(subD, X, seqLen, Y, batch_size=128, max_epochs=100, max_epochs_val=5, learning_rate=0.001)
    log_message += str(datetime.now()) + 'Finish training subD.\n'
    log_message += str(datetime.now()) + 'Training set result\t'
    log_message += score_template % subD.evaluate(np.hstack((X, np.zeros_like(X))), seqLen, Y)
    log_message += '\n' + str(datetime.now()) + 'Test set result\t'
    log_message += score_template % subD.evaluate(np.hstack((X_test, np.zeros_like(X_test))), seqLen_test, Y_test)
    log_writer(log_filepath, log_message)


if __name__ == '__main__':
    train_seq_malGAN()
