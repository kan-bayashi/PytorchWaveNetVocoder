#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2017 Tomoki Hayashi (Nagoya University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import numpy as np
import torch

from torch.autograd import Variable

from wavenet import initialize
from wavenet import UpSampling


def test_upsampling():
    aux = np.random.randn(1, 28, 1000)
    conv = UpSampling(10)
    conv.apply(initialize)
    batch = Variable(torch.from_numpy(aux).float())
    out = conv(batch)
    out = out.data.numpy()
    assert out.shape[-1] == aux.shape[-1] * 10
