from __future__ import division
import numpy as np
import torch
from torch.autograd import Variable
from wavenet import WaveNet, initialize, encode_mu_law, decode_mu_law


def sine_generator(seq_size=100, mu=256):
    t = np.linspace(0, 1, 16000)
    data = np.sin(2 * np.pi * 220 * t) + np.sin(2 * np.pi * 224 * t)
    data = data / 2
    while True:
        ys = data[:seq_size]
        ys = encode_mu_law(data, mu)
        yield Variable(torch.from_numpy(ys[:seq_size]).cuda())


net = WaveNet(256, 1, 16, 32, 10, 1, 2)
net.cuda()
net.apply(initialize)
generator = sine_generator(100)
batch = generator.next()
batch_input = batch.view(1, -1)
batch_aux = Variable(torch.FloatTensor(1, 1, batch_input.size(1)).fill_(0.0).cuda())
y = net(batch_input, batch_aux)
length = 100
batch_aux = Variable(torch.FloatTensor(1, 1, batch_input.size(1) + length).fill_(0.0).cuda(), volatile=True)
gen1 = net.generate(batch_input, batch_aux)
gen2 = net.fast_generate(batch_input, batch_aux)
wav1 = decode_mu_law(np.array(gen1), 256)
wav2 = decode_mu_law(np.array(gen2), 256)
