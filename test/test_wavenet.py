#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2017 Tomoki Hayashi (Nagoya University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

from __future__ import absolute_import

import logging

import numpy as np
import torch

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
        yield torch.from_numpy(ys[:seq_size])


def test_forward():
    # get batch
    generator = sine_generator(100)
    batch = next(generator)
    batch_input = batch.view(1, -1)
    batch_aux = torch.rand(1, 28, batch_input.size(1)).float()

    # define model without upsampling with kernel size = 2
    net = WaveNet(256, 28, 32, 128, 10, 1, 2)
    net.apply(initialize)
    net.eval()
    y = net(batch_input, batch_aux)[0]
    assert y.size(0) == batch_input.size(1)
    assert y.size(1) == 256

    # define model without upsampling with kernel size = 3
    net = WaveNet(256, 28, 32, 128, 10, 1, 2)
    net.apply(initialize)
    net.eval()
    y = net(batch_input, batch_aux)[0]
    assert y.size(0) == batch_input.size(1)
    assert y.size(1) == 256

    batch_input = batch.view(1, -1)
    batch_aux = torch.rand(1, 28, batch_input.size(1) // 10).float()

    # define model with upsampling and kernel size = 2
    net = WaveNet(256, 28, 32, 128, 10, 1, 2, 10)
    net.apply(initialize)
    net.eval()
    y = net(batch_input, batch_aux)[0]
    assert y.size(0) == batch_input.size(1)
    assert y.size(1) == 256

    # define model with upsampling and kernel size = 3
    net = WaveNet(256, 28, 32, 128, 10, 1, 3, 10)
    net.apply(initialize)
    net.eval()
    y = net(batch_input, batch_aux)[0]
    assert y.size(0) == batch_input.size(1)
    assert y.size(1) == 256


def test_generate():
    batch = 2
    x = np.random.randint(0, 256, size=(batch, 1))
    h = np.random.randn(batch, 28, 32)
    length = h.shape[-1] - 1
    with torch.no_grad():
        net = WaveNet(256, 28, 16, 32, 10, 3, 2)
        net.apply(initialize)
        net.eval()
        for x_, h_ in zip(x, h):
            batch_x = torch.from_numpy(np.expand_dims(x_, 0)).long()
            batch_h = torch.from_numpy(np.expand_dims(h_, 0)).float()
            net.generate(batch_x, batch_h, length, 1, "sampling")
            net.fast_generate(batch_x, batch_h, length, 1, "sampling")
        batch_x = torch.from_numpy(x).long()
        batch_h = torch.from_numpy(h).float()
        net.batch_fast_generate(batch_x, batch_h, [length] * batch, 1, "sampling")


def test_assert_fast_generation():
    # get batch
    batch = 2
    x = np.random.randint(0, 256, size=(batch, 1))
    h = np.random.randn(batch, 28, 100)
    length = h.shape[-1] - 1

    with torch.no_grad():
        # --------------------------------------------------------
        # define model without upsampling and with kernel size = 2
        # --------------------------------------------------------
        net = WaveNet(256, 28, 16, 32, 10, 3, 2)
        net.apply(initialize)
        net.eval()

        # sample-by-sample generation
        gen1_list = []
        gen2_list = []
        for x_, h_ in zip(x, h):
            batch_x = torch.from_numpy(np.expand_dims(x_, 0)).long()
            batch_h = torch.from_numpy(np.expand_dims(h_, 0)).float()
            gen1 = net.generate(batch_x, batch_h, length, 1, "argmax")
            gen2 = net.fast_generate(batch_x, batch_h, length, 1, "argmax")
            np.testing.assert_array_equal(gen1, gen2)
            gen1_list += [gen1]
            gen2_list += [gen2]
        gen1 = np.stack(gen1_list)
        gen2 = np.stack(gen2_list)
        np.testing.assert_array_equal(gen1, gen2)

        # batch generation
        batch_x = torch.from_numpy(x).long()
        batch_h = torch.from_numpy(h).float()
        gen3_list = net.batch_fast_generate(batch_x, batch_h, [length] * batch, 1, "argmax")
        gen3 = np.stack(gen3_list)
        np.testing.assert_array_equal(gen3, gen2)

        # --------------------------------------------------------
        # define model without upsampling and with kernel size = 3
        # --------------------------------------------------------
        net = WaveNet(256, 28, 16, 32, 10, 3, 3)
        net.apply(initialize)
        net.eval()

        # sample-by-sample generation
        gen1_list = []
        gen2_list = []
        for x_, h_ in zip(x, h):
            batch_x = torch.from_numpy(np.expand_dims(x_, 0)).long()
            batch_h = torch.from_numpy(np.expand_dims(h_, 0)).float()
            gen1 = net.generate(batch_x, batch_h, length, 1, "argmax")
            gen2 = net.fast_generate(batch_x, batch_h, length, 1, "argmax")
            np.testing.assert_array_equal(gen1, gen2)
            gen1_list += [gen1]
            gen2_list += [gen2]
        gen1 = np.stack(gen1_list)
        gen2 = np.stack(gen2_list)
        np.testing.assert_array_equal(gen1, gen2)

        # batch generation
        batch_x = torch.from_numpy(x).long()
        batch_h = torch.from_numpy(h).float()
        gen3_list = net.batch_fast_generate(batch_x, batch_h, [length] * batch, 1, "argmax")
        gen3 = np.stack(gen3_list)
        np.testing.assert_array_equal(gen3, gen2)

        # get batch
        batch = 2
        upsampling_factor = 10
        x = np.random.randint(0, 256, size=(batch, 1))
        h = np.random.randn(batch, 28, 10)
        length = h.shape[-1] * upsampling_factor - 1

        # -----------------------------------------------------
        # define model with upsampling and with kernel size = 2
        # -----------------------------------------------------
        net = WaveNet(256, 28, 16, 32, 10, 3, 2, upsampling_factor)
        net.apply(initialize)
        net.eval()

        # sample-by-sample generation
        gen1_list = []
        gen2_list = []
        for x_, h_ in zip(x, h):
            batch_x = torch.from_numpy(np.expand_dims(x_, 0)).long()
            batch_h = torch.from_numpy(np.expand_dims(h_, 0)).float()
            gen1 = net.generate(batch_x, batch_h, length, 1, "argmax")
            gen2 = net.fast_generate(batch_x, batch_h, length, 1, "argmax")
            np.testing.assert_array_equal(gen1, gen2)
            gen1_list += [gen1]
            gen2_list += [gen2]
        gen1 = np.stack(gen1_list)
        gen2 = np.stack(gen2_list)
        np.testing.assert_array_equal(gen1, gen2)

        # batch generation
        batch_x = torch.from_numpy(x).long()
        batch_h = torch.from_numpy(h).float()
        gen3_list = net.batch_fast_generate(batch_x, batch_h, [length] * batch, 1, "argmax")
        gen3 = np.stack(gen3_list)
        np.testing.assert_array_equal(gen3, gen2)

        # -----------------------------------------------------
        # define model with upsampling and with kernel size = 3
        # -----------------------------------------------------
        net = WaveNet(256, 28, 16, 32, 10, 3, 2, upsampling_factor)
        net.apply(initialize)
        net.eval()

        # sample-by-sample generation
        gen1_list = []
        gen2_list = []
        for x_, h_ in zip(x, h):
            batch_x = torch.from_numpy(np.expand_dims(x_, 0)).long()
            batch_h = torch.from_numpy(np.expand_dims(h_, 0)).float()
            gen1 = net.generate(batch_x, batch_h, length, 1, "argmax")
            gen2 = net.fast_generate(batch_x, batch_h, length, 1, "argmax")
            np.testing.assert_array_equal(gen1, gen2)
            gen1_list += [gen1]
            gen2_list += [gen2]
        gen1 = np.stack(gen1_list)
        gen2 = np.stack(gen2_list)
        np.testing.assert_array_equal(gen1, gen2)

        # batch generation
        batch_x = torch.from_numpy(x).long()
        batch_h = torch.from_numpy(h).float()
        gen3_list = net.batch_fast_generate(batch_x, batch_h, [length] * batch, 1, "argmax")
        gen3 = np.stack(gen3_list)
        np.testing.assert_array_equal(gen3, gen2)


def test_assert_different_length_batch_generation():
    # prepare batch
    batch = 4
    length = 100
    x = np.random.randint(0, 256, size=(batch, 1))
    h = np.random.randn(batch, 28, length)
    length_list = sorted(list(np.random.randint(length // 2, length - 1, batch)))

    with torch.no_grad():
        net = WaveNet(256, 28, 16, 32, 10, 3, 2)
        net.apply(initialize)
        net.eval()

        # sample-by-sample generation
        gen1_list = []
        for x_, h_, length in zip(x, h, length_list):
            batch_x = torch.from_numpy(np.expand_dims(x_, 0)).long()
            batch_h = torch.from_numpy(np.expand_dims(h_, 0)).float()
            gen1 = net.fast_generate(batch_x, batch_h, length, 1, "argmax")
            gen1_list += [gen1]

        # batch generation
        batch_x = torch.from_numpy(x).long()
        batch_h = torch.from_numpy(h).float()
        gen2_list = net.batch_fast_generate(batch_x, batch_h, length_list, 1, "argmax")

        # assertion
        for gen1, gen2 in zip(gen1_list, gen2_list):
            np.testing.assert_array_equal(gen1, gen2)
