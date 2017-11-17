#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time    : 17/11/17 下午4:11
# Author  : Shi Bo
# Email   : pkushibo@pku.edu.cn
# File    : networks.py


class blackboxDiscriminator():
    """
    black box Discrimanator
    """
    def __init__(self, cell_type='LSTM', rnn_layers=[128], ):