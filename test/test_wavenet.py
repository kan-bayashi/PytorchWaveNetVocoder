#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2017 Tomoki Hayashi (Nagoya University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

from __future__ import absolute_import

import logging

import numpy as np
import torch
from torch.autograd import Variable

from wavenet import encode_mu_law
from wavenet import initialize
from wavenet import WaveNet

# set log level
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S')


def sine_generator(seq_size=100, mu=256):
    t = np.linspace(0, 1, 16000)
    data = np.sin(2 * np.pi * 220 * t) + np.sin(2 * np.pi * 224 * t)
    data = data / 2
    while True:
        ys = data[:seq_size]
        ys = encode_mu_law(data, mu)
        yield Variable(torch.from_numpy(ys[:seq_size]))


def test_forward():
    # get batch
    generator = sine_generator(100)
    batch = next(generator)
    batch_input = batch.view(1, -1)
    batch_aux = Variable(torch.rand(1, 28, batch_input.size(1)).float())

    # define model without upsampling with kernel size = 2
    net = WaveNet(256, 28, 32, 128, 10, 1, 2)
    net.apply(initialize)
    net.eval()
    y = net(batch_input, batch_aux)
    assert y.size(0) == batch_input.size(1)
    assert y.size(1) == 256

    # define model without upsampling with kernel size = 3
    net = WaveNet(256, 28, 32, 128, 10, 1, 2)
    net.apply(initialize)
    net.eval()
    y = net(batch_input, batch_aux)
    assert y.size(0) == batch_input.size(1)
    assert y.size(1) == 256

    batch_input = batch.view(1, -1)
    batch_aux = Variable(torch.rand(1, 28, batch_input.size(1) // 10).float())

    # define model with upsampling and kernel size = 2
    net = WaveNet(256, 28, 32, 128, 10, 1, 2, 10)
    net.apply(initialize)
    net.eval()
    y = net(batch_input, batch_aux)
    assert y.size(0) == batch_input.size(1)
    assert y.size(1) == 256

    # define model with upsampling and kernel size = 3
    net = WaveNet(256, 28, 32, 128, 10, 1, 3, 10)
    net.apply(initialize)
    net.eval()
    y = net(batch_input, batch_aux)
    assert y.size(0) == batch_input.size(1)
    assert y.size(1) == 256


def test_generate():
    # get batch
    length = 100
    generator = sine_generator(100)
    batch = next(generator)
    batch_input = batch.view(1, -1)
    batch_aux = Variable(torch.rand(1, 28, batch_input.size(1) + length).float())

    # define model without upsampling and with kernel size = 2
    net = WaveNet(256, 28, 16, 32, 10, 3, 2)
    net.apply(initialize)
    net.cpu()
    net.eval()
    gen1 = net.generate(batch_input, batch_aux, length, 1, "argmax")
    gen2 = net.fast_generate(batch_input, batch_aux, length, 1, "argmax")
    np.testing.assert_array_equal(gen1, gen2)

    # define model without upsampling and with kernel size = 3
    net = WaveNet(256, 28, 16, 32, 10, 3, 3)
    net.apply(initialize)
    net.cpu()
    net.eval()
    gen1 = net.generate(batch_input, batch_aux, length, 1, "argmax")
    gen2 = net.fast_generate(batch_input, batch_aux, length, 1, "argmax")
    np.testing.assert_array_equal(gen1, gen2)

    batch_input = batch.view(1, -1)
    batch_aux = Variable(
        torch.rand(1, 28, (batch_input.size(1) + length) // 10 * 2).float())

    # define model with upsampling and kernel size = 2
    net = WaveNet(256, 28, 16, 32, 10, 3, 2, 10)
    net.apply(initialize)
    net.cpu()
    net.eval()
    gen1 = net.generate(batch_input, batch_aux, length, 1, "argmax")
    gen2 = net.fast_generate(batch_input, batch_aux, length, 1, "argmax")
    np.testing.assert_array_equal(gen1, gen2)

    # define model with upsampling and kernel size = 3
    net = WaveNet(256, 28, 16, 32, 10, 3, 3, 10)
    net.apply(initialize)
    net.cpu()
    net.eval()
    gen1 = net.generate(batch_input, batch_aux, length, 1, "argmax")
    gen2 = net.fast_generate(batch_input, batch_aux, length, 1, "argmax")
    np.testing.assert_array_equal(gen1, gen2)
