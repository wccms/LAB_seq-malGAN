#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time    : 2017/11/8 20:10
# Author  : Shi Bo
# File    : Networks.py

import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


class seqDiscriminator(nn.module):
    """
    seqDiscriminator: network to classify a sequence constructed from torch.nn.module
    """

    def __init__(self, vocab_num=160, embedding_dim=160, hidden_dim=128, is_bidirectional=False,
                 max_seq_len=1024, attention_layers=None, ff_layers=[512], class_num=2):
        """

        :param vocab_num: distinct feature numbers
        :param embedding_dim: embedding dimension from feature
        :param hidden_dim: dimension of hidden state
        :param is_bidirectional: indicating whether or not to use bidirectional rnn
        :param attention_layers: None or list of integers indicating layers of attention network
        :param ff_layers: list of integers indicating layers of feed forward network
        :param class_num: number of classification result
        :param max_seq_len: maximum length of a sequence
        """
        super(seqDiscriminator, self).__init__()
        # set hyper parameters
        self.vocab_num = vocab_num
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.is_bidirectional = is_bidirectional
        self.max_seq_len = max_seq_len
        self.attention_layers = attention_layers
        self.ff_layers = ff_layers
        self.class_num = class_num

        # define nn elements
        self.embedding = nn.Embedding(vocab_num, embedding_dim)
        self.LSTM = nn.LSTM(embedding_dim, hidden_dim, bidirectional=is_bidirectional)
        self.attnNet = []
        if attention_layers is not None:
            last_dim = hidden_dim * 2 if is_bidirectional else hidden_dim
            for i in range(attention_layers.size()):
                self.attnNet.append(nn.Linear(last_dim, attention_layers[i]))
                last_dim = attention_layers[i]
            self.attnNet.append(nn.Linear(last_dim, 1))
        self.ffNet = []
        last_dim = hidden_dim * 2 if is_bidirectional else hidden_dim
        for i in range(ff_layers.size()):
            self.ffNet.append(nn.Linear(last_dim, ff_layers[i]))
            last_dim = ff_layers[i]
        self.ffNet.append(nn.Linear(last_dim, class_num))

    def init_hidden(self, batch_size):
        if self.is_bidirectional:
            h = Variable(torch.randn(2, batch_size, self.hidden_dim))
            c = Variable(torch.randn(2, batch_size, self.hidden_dim))
        else:
            h = Variable(torch.randn(1, batch_size, self.hidden_dim))
            c = Variable(torch.randn(1, batch_size, self.hidden_dim))
        return h, c

    def forward(self, input, input_len, hidden0, cell0):
        # calculate forward output
        # input: batch_size * seq_len
        ouput = self.embedding(input)   # size: batch_size * seq_len * embedding_dim
        ouput = ouput.permute(1, 0, 2)  # size: seq_len * batch_size * embedding_dim
        h0, c0 = self.init_hidden(input.size()[0])
        output, _ = self.LSTM(ouput, h0, c0)    # size: seq_len * batch_size * (hidden_dim*direction)
        if self.attnNet.size() > 0:
            attn_weight = output
            for i in range(self.attnNet.size()):
                attn_weight = self.attnNet(output)  # size: seq_len * batch_size * 1
            attn_weight = torch.exp(attn_weight.squeeze())  # size: seq_len * batch_size
            attn_weight_sum = torch.sum(attn_weight, 0) # size: batch_size
            attn_weight = attn_weight / attn_weight_sum # size: seq_len * batch_size
            output = output * attn_weight.unsqueeze(-1)    # size: seq_len * batch_size * (hidden_dim*direction)
            output = torch.sum(output, 0)   # size: batch_size * (hidden_dim*direction)
        else:
            pass





