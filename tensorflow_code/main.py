#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time    : 2017/11/18 18:24
# Author  : Shi Bo
# File    : main.py

import numpy as np
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from datetime import datetime
import sys
import shutil
from tensorflow_code.networks import blackboxDiscriminator

from utils import load_dataset
from utils import write_log
from utils import dataLoader


def train_seq_malGAN():
    """
    main training function: first train subD, then alternately train boxD and malG
    :return: None
    """
    max_seq_len = 1024

    # make workspace directory for current mission and copy code
    timeTag = datetime.now().strftime('%Y-%m-%d_%H:%M')
    dir_path = '../tensorflow_result/'
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    dir_path = '../tensorflow_result/' + timeTag
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    if os.path.exists(os.path.join(dir_path, 'code')):
        shutil.rmtree(os.path.join(dir_path, 'code'))
    shutil.copytree('.', os.path.join(dir_path, 'code'))
    # log_filepath = dir_path + 'log.txt'
    score_template = 'TPR %(TPR)f\tFPR %(FPR)f\tAccuracy %(Accuracy)f\tAUC %(AUC)f'
    print(str(datetime.now()) + 'Start training seq_malGAN.')

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
    print(str(datetime.now()) + '\tFinish loading data.')

    # define substituteD as subD, black box D as boxD and malware Genarator as G
    subD = blackboxDiscriminator(cell_type='LSTM', rnn_layers=[128], is_bidirectionaal=False,
                                 attention_layers=[128], ff_layers=[128], batch_size=128, num_token=161,
                                 max_seq_len=2048, num_class=2, learning_rate=0.001, scope='blackboxD')
    # boxD_params = {'vocab_num': 160, 'embedding_dim': 160, 'hidden_dim': 128, 'is_bidirectional': False,
    #                'max_seq_len': 1024, 'attention_layers': None, 'ff_layers': [512], 'class_num': 2}
    # G_params = {}
    print(str(datetime.now()) + '\tFinish defining subD, boxD and G.')

    # train substitute Discrimanator first
    log_message = str(datetime.now()) + 'Start training black box Discriminator.'
    data_loader = dataLoader(X, seqLen, Y)
    subD.train(X, seqLen, Y, batch_size=128, max_epochs=100, max_epochs_val=5)
    print(str(datetime.now()) + 'Finish training subD.')
    print(str(datetime.now()) + 'Training set result:')
    print(score_template % subD.evaluate(np.hstack((X, np.zeros_like(X))), seqLen, Y))
    print(str(datetime.now()) + 'Test set result:')
    print(score_template % subD.evaluate(np.hstack((X_test, np.zeros_like(X_test))), seqLen_test, Y_test))

    # train substitute Discriminator and Generator of malGAN
    for epoch_i in range(5):
        pass
        # train G
        # todo

        # train D
        # todo

        # sample from G and evaluate on current black box D
        # todo

        # retrain black box D and evaluate generated data from G
        # todo

        # write to log
        # todo


if __name__ == '__main__':
    train_seq_malGAN()
