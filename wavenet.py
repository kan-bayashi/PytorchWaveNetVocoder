# -*- coding: utf-8 -*-
from __future__ import division

import logging
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable


def encode_mu_law(x, mu=256):
    """FUNCTION TO PERFORM MU-LAW ENCODING"""
    mu = mu - 1
    fx = np.sign(x) * np.log(1 + mu * np.abs(x)) / np.log(1 + mu)
    return np.floor((fx + 1) / 2 * mu + 0.5).astype(np.int64)


def decode_mu_law(y, mu=256):
    """FUNCTION TO PERFORM MU-LAW DECODING"""
    mu = mu - 1
    fx = (y - 0.5) / mu * 2 - 1
    x = np.sign(fx) / mu * ((1 + mu)**np.abs(fx) - 1)
    return x


def initialize(m):
    """FUCNTION TO INITILIZE CONV WITH XAVIER"""
    if isinstance(m, nn.Conv1d):
        nn.init.xavier_uniform(m.weight)
        nn.init.constant(m.bias, 0.0)


class OneHot(nn.Module):
    """CONVERT TO ONE-HOT VECTOR"""
    def __init__(self, depth):
        super(OneHot, self).__init__()
        self.depth = depth
        if torch.cuda.is_available():
            self.ones = torch.eye(depth).cuda()
        else:
            self.ones = torch.eye(depth)

    def forward(self, x):
        x_onehot = self.ones.index_select(0, x.data[0]).unsqueeze(0)
        if self.training:
            return Variable(x_onehot)
        else:
            return Variable(x_onehot, volatile=True)


class CausalConv1d(nn.Module):
    """1D DILATED CAUSAL CONVOLUTION"""
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, bias=True):
        super(CausalConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              padding=padding, dilation=dilation, bias=bias)

    def forward(self, x):
        x = self.conv(x)
        if self.padding != 0:
            x = x[:, :, :-self.padding]
        return x


class WaveNet(nn.Module):
    """CONDITIONAL WAVENET

    Args:
        n_quantize (int): number of quantization
        n_aux (int): number of aux feature dimension
        n_resch (int): number of filter channels for residual block
        n_skipch (int): number of filter channels for skip connection
        dilation_depth (int): number of dilation depth (e.g. if set 10, max dilation = 2**(10-1))
        dilation_repeat (int): number of dilation repeat
        kernel_size (int): filter size of dilated causal convolution

    """
    def __init__(self, n_quantize=256, n_aux=28, n_resch=512, n_skipch=256,
                 dilation_depth=10, dilation_repeat=3, kernel_size=2):
        super(WaveNet, self).__init__()
        self.n_aux = n_aux
        self.n_quantize = n_quantize
        self.n_resch = n_resch
        self.n_skipch = n_skipch
        self.kernel_size = kernel_size
        self.dilation_depth = dilation_depth
        self.dilation_repeat = dilation_repeat
        self.dilations = dilations = [2**i for i in range(dilation_depth)] * dilation_repeat
        self.receptive_field = (kernel_size - 1) * sum(dilations) + 1
        self.onehot = OneHot(n_quantize)
        self.causal = CausalConv1d(n_quantize, n_resch, kernel_size)
        self.dil_sigmoid = nn.ModuleList([CausalConv1d(n_resch, n_resch, kernel_size, d) for d in dilations])
        self.dil_tanh = nn.ModuleList([CausalConv1d(n_resch, n_resch, kernel_size, d) for d in dilations])
        self.aux_1x1_sigmoid = nn.ModuleList([nn.Conv1d(n_aux, n_resch, 1) for d in dilations])
        self.aux_1x1_tanh = nn.ModuleList([nn.Conv1d(n_aux, n_resch, 1) for d in dilations])
        self.skip_1x1 = nn.ModuleList([nn.Conv1d(n_resch, n_skipch, 1) for d in dilations])
        self.res_1x1 = nn.ModuleList([nn.Conv1d(n_resch, n_resch, 1) for d in dilations])
        self.conv_post_1 = nn.Conv1d(n_skipch, n_skipch, 1)
        self.conv_post_2 = nn.Conv1d(n_skipch, n_quantize, 1)

    def forward(self, x, h):
        output = self._preprocess(x)
        skip_connections = []
        for l in range(len(self.dilations)):
            output, skip = self._residual_forward(
                output, h, self.dil_sigmoid[l], self.dil_tanh[l],
                self.aux_1x1_sigmoid[l], self.aux_1x1_tanh[l],
                self.skip_1x1[l], self.res_1x1[l])
            skip_connections.append(skip)

        # sum up skip connections
        output = sum(skip_connections)
        output = self._postprocess(output)
        return output

    def _preprocess(self, x):
        x = self.onehot(x).transpose(1, 2)
        output = self.causal(x)
        return output

    def _postprocess(self, x):
        output = F.relu(x)
        output = self.conv_post_1(output)
        output = F.relu(output)
        output = self.conv_post_2(output).squeeze(0).transpose(0, 1)
        return output

    def _residual_forward(self, x, h, dil_sigmoid, dil_tanh,
                          aux_1x1_sigmoid, aux_1x1_tanh, skip_1x1, res_1x1):
        output_sigmoid = dil_sigmoid(x)
        output_tanh = dil_tanh(x)
        aux_output_sigmoid = aux_1x1_sigmoid(h)
        aux_output_tanh = aux_1x1_tanh(h)
        output = F.sigmoid(output_sigmoid + aux_output_sigmoid) * \
            F.tanh(output_tanh + aux_output_tanh)
        skip = skip_1x1(output)
        output = res_1x1(output)
        output = output + x
        return output, skip

    def generate(self, x, h, n_samples, intervals=None):
        # padding if the length less than receptive field size
        n_pad = self.receptive_field - x.size(1)
        if n_pad > 0:
            x = F.pad(x, (n_pad, 0), "constant", self.n_quantize // 2)
            h = F.pad(h, (n_pad, 0), "replicate")

        # generate
        samples = x.data[0].tolist()
        start = time.time()
        for i in range(n_samples):
            current_idx = len(samples)
            x = Variable(torch.LongTensor(samples[-self.receptive_field:]).view(1, -1).cuda(),
                         volatile=True)
            h_ = h[:, :, current_idx - self.receptive_field: current_idx]
            output = self.forward(x, h_)
            posterior = F.softmax(output[-1], dim=0)
            dist = torch.distributions.Categorical(posterior)
            sample = dist.sample().data[0]
            samples.append(sample)

            # show progress
            if intervals is not None and (i + 1) % intervals == 0:
                logging.info("%d/%d estimated time = %.3f sec (%.3f sec / sample)" % (
                    i + 1, n_samples,
                    (n_samples - i - 1) * ((time.time() - start) / (i + 1)),
                    (time.time() - start) / (i + 1)))

        return np.array(samples[-n_samples:])

    def fast_generate(self, x, h, n_samples, intervals=None):
        # padding if the length less than
        n_pad = self.receptive_field - x.size(1)
        if n_pad > 0:
            x = F.pad(x, (n_pad, 0), "constant", self.n_quantize // 2)
            h = F.pad(h, (n_pad, 0), "replicate")

        # prepare buffer
        output = self._preprocess(x)
        h_ = h[:, :, :x.size(1)]
        output_buffer = []
        buffer_size = []
        for l, d in enumerate(self.dilations):
            output, _ = self._residual_forward(
                output, h_, self.dil_sigmoid[l], self.dil_tanh[l],
                self.aux_1x1_sigmoid[l], self.aux_1x1_tanh[l],
                self.skip_1x1[l], self.res_1x1[l])
            if d == 2**(self.dilation_depth - 1):
                buffer_size.append(self.kernel_size - 1)
            else:
                buffer_size.append(d * 2 * (self.kernel_size - 1))
            output_buffer.append(output[:, :, -buffer_size[l] - 1: -1])

        # generate
        samples = x.data[0].tolist()
        start = time.time()
        for i in range(n_samples):
            output = Variable(torch.LongTensor(samples[-self.kernel_size * 2 - 1:]).view(1, -1).cuda(),
                              volatile=True)
            output = self._preprocess(output)
            h_ = h[:, :, len(samples) - 1].contiguous().view(1, h.size(1), 1)
            output_buffer_next = []
            skip_connections = []
            for l, d in enumerate(self.dilations):
                output, skip = self._residual_forward(
                    output, h_, self.dil_sigmoid[l], self.dil_tanh[l],
                    self.aux_1x1_sigmoid[l], self.aux_1x1_tanh[l],
                    self.skip_1x1[l], self.res_1x1[l])
                output = output[:, :, -1:]
                output = torch.cat([output_buffer[l], output], dim=2)
                skip_connections.append(skip[:, :, -1:])
                output_buffer_next.append(output[:, :, -buffer_size[l]:])

            # update buffer
            output_buffer = output_buffer_next

            # calculate posterior
            output = sum(skip_connections)
            output = self._postprocess(output)
            posterior = F.softmax(output[-1], dim=0)

            # perform sampling
            dist = torch.distributions.Categorical(posterior)
            sample = dist.sample().data.cpu().numpy()[0]
            samples.append(sample)

            # show progress
            if intervals is not None and (i + 1) % intervals == 0:
                logging.info("%d/%d estimated time = %.3f sec (%.3f sec / sample)" % (
                    i + 1, n_samples,
                    (n_samples - i - 1) * ((time.time() - start) / (i + 1)),
                    (time.time() - start) / (i + 1)))

        return np.array(samples[-n_samples:])

    def faster_generate(self, x, h, n_samples, intervals=None):
        # padding if the length less than
        n_pad = self.receptive_field - x.size(1)
        if n_pad > 0:
            x = F.pad(x, (n_pad, 0), "constant", self.n_quantize // 2)
            h = F.pad(h, (n_pad, 0), "replicate")

        # prepare buffer
        output = self._preprocess(x)
        h_ = h[:, :, :x.size(1)]
        output_buffer = []
        buffer_size = []
        for l, d in enumerate(self.dilations):
            output, _ = self._residual_forward(
                output, h_, self.dil_sigmoid[l], self.dil_tanh[l],
                self.aux_1x1_sigmoid[l], self.aux_1x1_tanh[l],
                self.skip_1x1[l], self.res_1x1[l])
            if d == 2**(self.dilation_depth - 1):
                buffer_size.append(self.kernel_size - 1)
            else:
                buffer_size.append(d * 2 * (self.kernel_size - 1))
            output_buffer.append(output[:, :, -buffer_size[l] - 1: -1])

        # generate
        samples = x.data[0]
        start = time.time()
        for i in range(n_samples):
            output = Variable(samples[-self.kernel_size * 2 - 1:].unsqueeze(0), volatile=True)
            output = self._preprocess(output)
            h_ = h[:, :, len(samples) - 1].contiguous().view(1, self.n_aux, 1)
            output_buffer_next = []
            skip_connections = []
            for l, d in enumerate(self.dilations):
                output, skip = self._generate_residual_forward(
                    output, h_, self.dil_sigmoid[l], self.dil_tanh[l],
                    self.aux_1x1_sigmoid[l], self.aux_1x1_tanh[l],
                    self.skip_1x1[l], self.res_1x1[l])
                output = torch.cat([output_buffer[l], output], dim=2)
                output_buffer_next.append(output[:, :, -buffer_size[l]:])
                skip_connections.append(skip)

            # update buffer
            output_buffer = output_buffer_next

            # calculate posterior
            output = sum(skip_connections)
            output = self._postprocess(output)
            posterior = F.softmax(output[-1], dim=0)

            # perform sampling
            dist = torch.distributions.Categorical(posterior)
            sample = dist.sample().data
            samples = torch.cat([samples, sample], dim=0)

            # show progress
            if intervals is not None and (i + 1) % intervals == 0:
                logging.info("%d/%d estimated time = %.3f sec (%.3f sec / sample)" % (
                    i + 1, n_samples,
                    (n_samples - i - 1) * ((time.time() - start) / (i + 1)),
                    (time.time() - start) / (i + 1)))

        return samples[-n_samples:].cpu().numpy()

    def _generate_residual_forward(self, x, h, dil_sigmoid, dil_tanh,
                                   aux_1x1_sigmoid, aux_1x1_tanh, skip_1x1, res_1x1):
        output_sigmoid = dil_sigmoid(x)[:, :, -1:]
        output_tanh = dil_tanh(x)[:, :, -1:]
        aux_output_sigmoid = aux_1x1_sigmoid(h)
        aux_output_tanh = aux_1x1_tanh(h)
        output = F.sigmoid(output_sigmoid + aux_output_sigmoid) * \
            F.tanh(output_tanh + aux_output_tanh)
        skip = skip_1x1(output)
        output = res_1x1(output)
        output = output + x[:, :, -1:]
        return output, skip
