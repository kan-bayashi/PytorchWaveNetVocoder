#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division

import argparse
import logging
import os
import sys
import time

import numpy as np
import six
import soundfile as sf
import torch
from dateutil.relativedelta import relativedelta
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.autograd import Variable
from torchvision import transforms

from utils import background, find_files, read_hdf5
from wavenet import WaveNet, encode_mu_law, initialize


def validate_length(x, y):
    """FUNCTION TO VALIDATE LENGTH

    Args:
        x (ndarray): numpy.ndarray with x.shape[0] = len_x
        y (ndarray): numpy.ndarray with y.shape[0] = len_y

    Returns:
        x with x.shape[0] = min(len_x, len_y)
        y with y.shape[0] = min(len_x, len_y)

    """
    if x.shape[0] < y.shape[0]:
        y = y[:x.shape[0]]
    if x.shape[0] > y.shape[0]:
        x = x[:y.shape[0]]
    assert len(x) == len(y)

    return x, y


@background(max_prefetch=16)
def custom_generator(wavdir, featdir, receptive_field=None, batch_size=None,
                     wav_transform=None, feat_transform=None, shuffle=True,
                     use_speaker_code=False):
    """TRAINING BATCH GENERATOR

    Args:
        wavdir (str): directory including wav files
        featdir (str): directory including feat files
        receptive_field (int): size of receptive filed
        batch_size (int): batch size
        wav_transform (func): preprocessing function for waveform
        feat_transform (func): preprocessing function for aux feats
        shuffle (bool): whether to do shuffle of the file list
        use_speaker_code (bool): whether to use speaker code

    Return: generator instance

    """
    # get file list
    filenames = sorted(find_files(wavdir, "*.wav", use_dir_name=False))
    wav_list = [wavdir + "/" + filename for filename in filenames]
    feat_list = [featdir + "/" + filename.replace(".wav", ".h5") for filename in filenames]

    # shuffle list
    if shuffle:
        n_files = len(wav_list)
        idx = np.random.permutation(n_files)
        wav_list = [wav_list[i] for i in idx]
        feat_list = [feat_list[i] for i in idx]

    # generator part
    while True:
        for wavfile, featfile in zip(wav_list, feat_list):
            x, fs = sf.read(wavfile, dtype=np.float32)
            h = read_hdf5(featfile, "/feat")
            if use_speaker_code:
                sc = read_hdf5(featfile, "/speaker_code")
                sc = np.tile(sc, [h.shape[0], 1])
                h = np.concatenate([h, sc], axis=1)

            # check both lengths are same
            x, h = validate_length(x, h)

            # cut utterance into small batch
            if batch_size is not None:
                if "x_buffer" not in locals():
                    x_buffer = np.empty((0), dtype=np.float32)
                    h_buffer = np.empty((0, h.shape[1]), dtype=np.float32)
                x_buffer = np.concatenate([x_buffer, x], axis=0)
                h_buffer = np.concatenate([h_buffer, h], axis=0)

                while len(x_buffer) > receptive_field + batch_size:
                    x_ = x_buffer[:receptive_field + batch_size]
                    h_ = h_buffer[:receptive_field + batch_size]

                    if wav_transform is not None:
                        x_ = wav_transform(x_)
                    if feat_transform is not None:
                        h_ = feat_transform(h_)

                    batch_x = x_[:-1].unsqueeze(0)
                    batch_h = h_[:-1].transpose(0, 1).unsqueeze(0)
                    batch_target = x_[1:]

                    yield (batch_x, batch_h), batch_target

                    x_buffer = x_buffer[batch_size:]
                    h_buffer = h_buffer[batch_size:]

            # utterance batch
            else:
                if wav_transform is not None:
                    x = wav_transform(x)
                if feat_transform is not None:
                    h = feat_transform(h)

                batch_x = x[:-1].unsqueeze(0)
                batch_h = h[:-1].transpose(0, 1).unsqueeze(0)
                batch_target = x[1:]

                yield (batch_x, batch_h), batch_target

        # re-shuffle
        if shuffle:
            idx = np.random.permutation(n_files)
            wav_list = [wav_list[i] for i in idx]
            feat_list = [feat_list[i] for i in idx]


def save_checkpoint(checkpoint_dir, model, optimizer, iterations):
    """FUNCTION TO SAVE CHECKPOINT

    Args:
        checkpoint_dir (str): directory to save checkpoint
        model (torch.nn.Module): pytorch model instance
        optimizer (Optimizer): pytorch optimizer instance
        iterations (int): number of current iterations

    """
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iterations": iterations}
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    torch.save(checkpoint, checkpoint_dir + "/checkpoint-%d.pkl" % iterations)
    logging.info("%d-iter checkpoint created." % iterations)


def main():
    parser = argparse.ArgumentParser()
    # path setting
    parser.add_argument("--wavdir", required=True,
                        type=str, help="directory including wav files")
    parser.add_argument("--featdir", required=True,
                        type=str, help="directory including aux feat files")
    parser.add_argument("--expdir", required=True,
                        type=str, help="directory to save the model")
    parser.add_argument("--stats", required=True,
                        type=str, help="hdf5 file including statistics")
    # network structure setting
    parser.add_argument("--n_quantize", default=256,
                        type=int, help="number of quantization")
    parser.add_argument("--n_aux", default=28,
                        type=int, help="number of dimension of aux feats")
    parser.add_argument("--n_resch", default=512,
                        type=int, help="number of channels of residual output")
    parser.add_argument("--n_skipch", default=256,
                        type=int, help="number of channels of skip output")
    parser.add_argument("--dilation_depth", default=10,
                        type=int, help="depth of dilation")
    parser.add_argument("--dilation_repeat", default=3,
                        type=int, help="number of repeating of dilation")
    parser.add_argument("--kernel_size", default=2,
                        type=int, help="kerne size of dilated causal convolution")
    # network training setting
    parser.add_argument("--lr", default=1e-3,
                        type=float, help="learning rate")
    parser.add_argument("--weight_decay", default=0.0,
                        type=float, help="weight decay coefficient")
    parser.add_argument("--batch_size", default=20000,
                        type=int, help="number of iterations")
    parser.add_argument("--n_iter", default=200000,
                        type=int, help="number of iterations")
    # other setting
    parser.add_argument("--n_checkpoint", default=25000,
                        type=int, help="how frequent saving model")
    parser.add_argument("--n_interval", default=1000,
                        type=int, help="log interval")
    parser.add_argument("--seed", default=1,
                        type=int, help="seed number")
    parser.add_argument("--resume", default=None,
                        type=str, help="model path to restart training")
    parser.add_argument("--verbose", default=1,
                        type=int, help="log level")
    args = parser.parse_args()

    # set log level
    if args.verbose > 0:
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S')
    elif args.verbose > 1:
        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S')
    else:
        logging.basicConfig(level=logging.WARN,
                            format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S')
        logging.warn("logging is disabled.")

    # show arguments and save args as conf
    if not os.path.exists(args.expdir):
        os.makedirs(args.expdir)
    torch.save(args, args.expdir + "/model.conf")

    # define network
    model = WaveNet(n_quantize=args.n_quantize,
                    n_aux=args.n_aux,
                    n_resch=args.n_resch,
                    n_skipch=args.n_skipch,
                    dilation_depth=args.dilation_depth,
                    dilation_repeat=args.dilation_repeat,
                    kernel_size=args.kernel_size)
    logging.info(model)
    model.apply(initialize)

    # define loss and optimizer
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    # send to gpu
    if torch.cuda.is_available():
        model.cuda()
        criterion.cuda()
    else:
        logging.error("gpu is not available. please check the setting.")
        sys.exit(1)

    # define transforms
    scaler = StandardScaler()
    scaler.mean_ = read_hdf5(args.stats, "/mean")
    scaler.scale_ = read_hdf5(args.stats, "/scale")
    wav_transform = transforms.Compose([
        lambda x: encode_mu_law(x, args.n_quantize),
        lambda x: Variable(torch.from_numpy(x).long().cuda())])
    feat_transform = transforms.Compose([
        lambda x: scaler.transform(x),
        lambda x: Variable(torch.from_numpy(x).float().cuda())])

    # define generator
    generator = custom_generator(
        args.wavdir, args.featdir, model.receptive_field, args.batch_size,
        wav_transform, feat_transform, True, False)

    # resume
    if args.resume is not None:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        iterations = checkpoint["iterations"]
        logging.info("restored from %d-iter checkpoint." % iterations)
    else:
        iterations = 0

    # train
    loss = 0
    total = 0
    for i in six.moves.range(iterations, args.n_iter):
        batch_start = time.time()
        (batch_x, batch_h), batch_target = generator.next()
        batch_output = model(batch_x, batch_h)
        batch_loss = criterion(batch_output[model.receptive_field:],
                               batch_target[model.receptive_field:])
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        logging.info("batch loss = %.3f (time = %.3f / batch)" % (
            batch_loss.data[0], time.time()-batch_start))
        total += time.time() - batch_start
        loss += batch_loss.data[0]

        # report progress
        if (i + 1) % args.n_interval == 0:
            logging.info("(iter:%d) loss = %.6f (%.3f sec / batch)" % (
                i + 1, loss / args.n_interval, total / args.n_interval))
            logging.info("estimated required time = "
                         "{0.days:02}:{0.hours:02}:{0.minutes:02}:{0.seconds:02}"
                         .format(relativedelta(
                             seconds=int((args.n_iter - (i + 1)) * (total / args.n_interval)))))
            total = 0
            loss = 0

        # save intermidiate model
        if (i + 1) % args.n_checkpoint == 0:
            save_checkpoint(args.expdir, model, optimizer, i + 1)

    # save final model
    save_checkpoint(args.expdir, model, optimizer, args.n_iter)


if __name__ == "__main__":
    main()
