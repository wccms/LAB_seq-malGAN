#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time    : 17/11/7 下午4:33
# Author  : Shi Bo
# Email   : pkushibo@pku.edu.cn
# File    : train_seq_malGAN.py

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from datetime import datetime
import sys
import shutil
import numpy as np
from utils import load_dataset