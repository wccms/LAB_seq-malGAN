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
import torch.utils.data as Data

def train_D(D, X, seqLen, Y):
    """
    train a Discrimanator using X, seqLen and Y
    :param D:
    :param X:
    :param seqLen:
    :param Y:
    :return:
    """
    pass


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
    log_message = str(datetime.now()) + '\tStart training seq_malGAN.\n'

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
    subD_params = {'vocab_num':160, 'embedding_dim':160, 'hidden_dim':128, 'is_bidirectional':False,
                 'max_seq_len':2048, 'attention_layers':None, 'ff_layers':[512], 'class_num':2}
    # boxD_params = {'vocab_num': 160, 'embedding_dim': 160, 'hidden_dim': 128, 'is_bidirectional': False,
    #                'max_seq_len': 1024, 'attention_layers': None, 'ff_layers': [512], 'class_num': 2}
    # G_params = {}
    subD = seqDiscriminator(**subD_params)
    log_message += str(datetime.now()) + '\tFinish defining subD, boxD and G.\n'

    # train substitute Discrimanator first
    train_D(subD, X, seqLen, Y)
    log_message += str(datetime.now()) + '\tFinish training subD.\n'
    log_message += str(datetime.now()) + '\tTraining set result\t'
    log_message += score_template % subD.evaluate(np.hstack((X, np.zeros_like(X))), seqLen, Y)
    log_message += '\n' + str(datetime.now()) + '\tTest set result\t'
    log_message += score_template % subD.evaluate(np.hstack((X_test, np.zeros_like(X_test))), seqLen_test, Y_test)




if __name__ == '__main__':
    train_seq_malGAN()
