from __future__ import absolute_import

import logging

import numpy as np
import torch
from torch.autograd import Variable

from wavenet import WaveNet, initialize, encode_mu_law


def sine_generator(seq_size=100, mu=256):
    t = np.linspace(0, 1, 16000)
    data = np.sin(2 * np.pi * 220 * t) + np.sin(2 * np.pi * 224 * t)
    data = data / 2
    while True:
        ys = data[:seq_size]
        ys = encode_mu_law(data, mu)
        yield Variable(torch.from_numpy(ys[:seq_size]).cuda())


# set log level
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S')

# define model
net = WaveNet(256, 28, 512, 256, 10, 3, 2)
net.apply(initialize)
net.cuda()
net.eval()

# forward test
generator = sine_generator(100)
batch = next(generator)
batch_input = batch.view(1, -1)
batch_aux = Variable(torch.rand(1, 28, batch_input.size(1)).float().cuda())
y = net(batch_input, batch_aux)

# generation test
length = 100
batch_aux = Variable(torch.rand(1, 28, batch_input.size(1) + length).float().cuda(), volatile=True)
gen1 = net.generate(batch_input, batch_aux, length, 1, "argmax")
gen2 = net.fast_generate(batch_input, batch_aux, length, 1, "argmax")
np.testing.assert_array_equal(gen1, gen2)

# define model
net = WaveNet(256, 28, 512, 256, 10, 3, 2, 10)
net.apply(initialize)
net.cuda()
net.eval()

# forward test
generator = sine_generator(100)
batch = next(generator)
batch_input = batch.view(1, -1)
batch_aux = Variable(torch.rand(1, 28, 10).float().cuda())
y = net(batch_input, batch_aux)

# generation test
length = 100
batch_aux = Variable(torch.rand(1, 28, 20).float().cuda(), volatile=True)
gen1 = net.generate(batch_input, batch_aux, length, 1, "argmax")
gen2 = net.fast_generate(batch_input, batch_aux, length, 1, "argmax")
np.testing.assert_array_equal(gen1, gen2)
