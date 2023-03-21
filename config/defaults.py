# --------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# --------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from yacs.config import CfgNode as CN

_C = CN()

_C.MODEL = CN()
_C.MODEL.NAME = 'CoSOD'
_C.MODEL.PRETRAINED = ''

_C.MODEL.ENCODER = CN()
_C.MODEL.ENCODER.NAME = []
_C.MODEL.ENCODER.CHANNEL = []
_C.MODEL.ENCODER.STRIDE = []

_C.MODEL.DASPP = CN()
_C.MODEL.DASPP.ADAP_CHANNEL = 512
_C.MODEL.DASPP.DILATIONS = []

_C.MODEL.GROUP_ATTENTION = CN()
_C.MODEL.GROUP_ATTENTION.NAME = []
_C.MODEL.GROUP_ATTENTION.CHANNEL = []
_C.MODEL.GROUP_ATTENTION.NUM_HEADS = 8
_C.MODEL.GROUP_ATTENTION.DROP_RATE = 0.1
_C.MODEL.GROUP_ATTENTION.MSP_SCALES = []

_C.MODEL.COFORMER_DECODER = CN()
_C.MODEL.COFORMER_DECODER.HIDDEN_DIM = 256
_C.MODEL.COFORMER_DECODER.DROP_PATH = 0.1
_C.MODEL.COFORMER_DECODER.NUM_HEADS = 8
_C.MODEL.COFORMER_DECODER.FEEDFORWARD_DIM = 512
_C.MODEL.COFORMER_DECODER.FFN_EXP = 3
