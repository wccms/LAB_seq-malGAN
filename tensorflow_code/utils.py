#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time    : 17/11/7 下午4:35
# Author  : Shi Bo
# Email   : pkushibo@pku.edu.cn
# File    : utils.py

import numpy as np
from datetime import datetime
from sklearn import metrics

def load_dataset(data_path, max_seq_len=2048, pad_len=2048):
    """ utils: load the dataset
    :param data_path: the path of the dataset
    :param max_seq_len: the max length of a sequence
    :param pad_len: the padded length of the returned matrix
    :return: X_malware, malware_length, X_benign, benign_length
             X_malware and X_benign are the matrix of malware and benign.
                shape: [num instances, max_seq_len + pad_len]
             malware_length and benign_length are the length of the matrix, which should be smaller than max_seq_len
                shape: [num instances]
    """
    X_malware = []
    malware_length = []
    X_benign = []
    benign_length = []
    for line in open(data_path):
        elements = line.strip().split(';')
        Xi = []
        for element in elements[2:-1]:
            if len(element) <= 0:
                continue
            for digit in element.split(','):
                Xi.append(int(digit))
        Xi = Xi[:max_seq_len]
        if elements[1] is '0':
            benign_length.append(len(Xi))
            X_benign.append(np.array(Xi + [0] * (max_seq_len + pad_len - len(Xi)), dtype=np.int32))
        else:
            malware_length.append(len(Xi))
            X_malware.append(np.array(Xi + [0] * (max_seq_len + pad_len - len(Xi)), dtype=np.int32))
    return np.vstack(X_malware), np.array(malware_length), np.vstack(X_benign), np.array(benign_length)


def evaluate(model, X, seq_len, trueY):
    """
    evaluate a model
    :param model:
    :param X:
    :param seq_len:
    :return:
    """
    # print('in evaluate: len(X)=%d\tlen(seq_len)=%d\tlen(trueY)=%d'%(len(X),len(seq_len),len(trueY)))
    # indexes=np.arange(len(X))
    # np.random.shuffle(indexes)
    # X=X[indexes]
    # X=X[:1000]
    # seq_len=seq_len[indexes]
    # seq_len=seq_len[:1000]
    # trueY=trueY[indexes]
    # trueY=trueY[:1000]
    pred_proba = model.predict_proba(X, seq_len)[:, 1]
    score_dict = {}
    score_dict['AUC'] = metrics.roc_auc_score(trueY, pred_proba)
    predY = [0 if proba < 0.5 else 1 for proba in pred_proba]
    score_dict['Accuracy'] = metrics.accuracy_score(trueY, predY)
    confusionMatrix = metrics.confusion_matrix(trueY, predY)
    score_dict['ConfusionMatrix'] = confusionMatrix
    score_dict['TPR'] = confusionMatrix[1, 1] / float(confusionMatrix[1, 0] + confusionMatrix[1, 1])
    score_dict['FPR'] = confusionMatrix[0, 0] / float(confusionMatrix[0, 0] + confusionMatrix[0, 1])
    return score_dict


def write_log(log_filepath='log.txt', log_message=str(datetime.now())):
    """
    write log message to log file at log_filepath
    :param log_filepath: filepath to write log to
    :param log_message: log messages to write
    :return: None
    """
    with open(log_filepath, 'a') as f:
        f.write(log_message + '\n')


class dataLoader():
    """
    data loader
    """

    def __init__(self, X, seq_len, Y, shuffle=True):
        """
        initialize dataLoader
        :param X:
        :param seq_len:
        :param Y:
        """
        if shuffle:
            indexes = np.arange(len(X))
            np.random.shuffle(indexes)
            X = X[indexes]
            seq_len = seq_len[indexes]
            Y = Y[indexes]
        self.X = X
        self.seq_len = seq_len
        self.Y = Y

        self.index = 0
        self.data_num = len(X)

    def next_batch(self, batch_size=128):
        """
        return next batch of data
        :param batch_size:
        :return: next batch of data
        """
        if self.index + batch_size > self.data_num:
            X = self.X[self.index:]
            seq_len = self.seq_len[self.index:]
            Y = self.Y[self.index:]
            left_num = self.index + batch_size - self.data_num
            self.index = 0
            left_X, left_seq_len, left_Y = self.next_batch(left_num)
            return np.vstack((X, left_X)), np.vstack((seq_len, left_seq_len)), np.vstack((Y, left_Y))

        else:
            self.index += batch_size
            return self.X[self.index - batch_size:self.index], \
                   self.seq_len[self.index - batch_size:self.index], \
                   self.Y[self.index - batch_size:self.index],
