#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2017 Tomoki Hyaashi (Nagoya University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

from __future__ import print_function, division

import argparse
import logging
import os
import sys
import time
from distutils.util import strtobool

import numpy as np
import six
import soundfile as sf
import torch
from dateutil.relativedelta import relativedelta
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.autograd import Variable
from torchvision import transforms

from utils import background, find_files, read_hdf5, read_txt
from wavenet import WaveNet, encode_mu_law, initialize


def validate_length(x, y, upsampling_factor=0):
    """FUNCTION TO VALIDATE LENGTH

    Args:
        x (ndarray): numpy.ndarray with x.shape[0] = len_x
        y (ndarray): numpy.ndarray with y.shape[0] = len_y
        upsampling_factor (int): upsampling factor

    Returns:
        (ndarray): length adjusted x with same length y
        (ndarray): length adjusted y with same length x
    """
    if upsampling_factor == 0:
        if x.shape[0] < y.shape[0]:
            y = y[:x.shape[0]]
        if x.shape[0] > y.shape[0]:
            x = x[:y.shape[0]]
        assert len(x) == len(y)
    else:
        if x.shape[0] > y.shape[0] * upsampling_factor:
            x = x[:y.shape[0] * upsampling_factor]
        if x.shape[0] < y.shape[0] * upsampling_factor:
            mod_y = y.shape[0] * upsampling_factor - x.shape[0]
            mod_y_frame = mod_y // upsampling_factor + 1
            y = y[:-mod_y_frame]
            x = x[:y.shape[0] * upsampling_factor]
        assert len(x) == len(y) * upsampling_factor

    return x, y


@background(max_prefetch=16)
def train_generator(wav_list, feat_list, receptive_field, batch_size=0,
                    wav_transform=None, feat_transform=None, shuffle=True,
                    upsampling_factor=0, use_speaker_code=False):
    """TRAINING BATCH GENERATOR

    Args:
        wav_list (str): list of wav files
        feat_list (str): list of feat files
        receptive_field (int): size of receptive filed
        batch_size (int): batch size
        wav_transform (func): preprocessing function for waveform
        feat_transform (func): preprocessing function for aux feats
        shuffle (bool): whether to do shuffle of the file list
        upsampling_factor (int): upsampling factor
        use_speaker_code (bool): whether to use speaker code

    Return:
        (object): generator instance
    """
    # shuffle list
    if shuffle:
        n_files = len(wav_list)
        idx = np.random.permutation(n_files)
        wav_list = [wav_list[i] for i in idx]
        feat_list = [feat_list[i] for i in idx]

    # check batch_size
    if batch_size != 0 and upsampling_factor != 0:
        batch_mod = (receptive_field + batch_size) % upsampling_factor
        logging.warn("batch size is decreased due to upsampling (%d -> %d)" % (
            batch_size, batch_size - batch_mod))
        batch_size -= batch_mod

    while True:
        # process over all of files
        for wavfile, featfile in zip(wav_list, feat_list):
            # load wavefrom and aux feature
            x, fs = sf.read(wavfile, dtype=np.float32)
            if upsampling_factor > 0:
                h = read_hdf5(featfile, "/feat_org")
            else:
                h = read_hdf5(featfile, "/feat")
            if use_speaker_code:
                sc = read_hdf5(featfile, "/speaker_code")
                sc = np.tile(sc, [h.shape[0], 1])
                h = np.concatenate([h, sc], axis=1)

            # check both lengths are same
            logging.debug("before x length = %d" % x.shape[0])
            logging.debug("before h length = %d" % h.shape[0])
            x, h = validate_length(x, h, upsampling_factor)
            logging.debug("after x length = %d" % x.shape[0])
            logging.debug("after h length = %d" % h.shape[0])

            # use mini batch without upsampling
            if batch_size != 0 and upsampling_factor == 0:
                # make buffer array
                if "x_buffer" not in locals():
                    x_buffer = np.empty((0), dtype=np.float32)
                    h_buffer = np.empty((0, h.shape[1]), dtype=np.float32)
                x_buffer = np.concatenate([x_buffer, x], axis=0)
                h_buffer = np.concatenate([h_buffer, h], axis=0)

                while len(x_buffer) > receptive_field + batch_size:
                    # get pieces
                    x_ = x_buffer[:receptive_field + batch_size]
                    h_ = h_buffer[:receptive_field + batch_size]

                    # perform pre-processing
                    if wav_transform is not None:
                        x_ = wav_transform(x_)
                    if feat_transform is not None:
                        h_ = feat_transform(h_)

                    # remove the last and first sample for training
                    batch_x = x_[:-1].unsqueeze(0)
                    batch_h = h_[:-1].transpose(0, 1).unsqueeze(0)
                    batch_t = x_[1:]

                    yield (batch_x, batch_h), batch_t

                    # update buffer
                    x_buffer = x_buffer[batch_size:]
                    h_buffer = h_buffer[batch_size:]

            # use mini batch with upsampling
            elif batch_size != 0 and upsampling_factor > 0:
                # make buffer array
                if "x_buffer" not in locals():
                    x_buffer = np.empty((0), dtype=np.float32)
                    h_buffer = np.empty((0, h.shape[1]), dtype=np.float32)
                x_buffer = np.concatenate([x_buffer, x], axis=0)
                h_buffer = np.concatenate([h_buffer, h], axis=0)

                while len(h_buffer) > (receptive_field + batch_size) // upsampling_factor:
                    # set batch size
                    h_bs = (receptive_field + batch_size) // upsampling_factor
                    x_bs = h_bs * upsampling_factor + 1

                    # get pieces
                    h_ = h_buffer[:h_bs]
                    x_ = x_buffer[:x_bs]

                    # perform pre-processing
                    if wav_transform is not None:
                        x_ = wav_transform(x_)
                    if feat_transform is not None:
                        h_ = feat_transform(h_)

                    # remove the last and first sample for training
                    batch_h = h_.transpose(0, 1).unsqueeze(0)
                    batch_x = x_[:-1].unsqueeze(0)
                    batch_t = x_[1:]

                    yield (batch_x, batch_h), batch_t

                    # set shift size
                    h_ss = batch_size // upsampling_factor
                    x_ss = h_ss * upsampling_factor

                    # update buffer
                    h_buffer = h_buffer[h_ss:]
                    x_buffer = x_buffer[x_ss:]

            # use utterance batch without upsampling
            elif batch_size == 0 and upsampling_factor == 0:
                # perform pre-processing
                if wav_transform is not None:
                    x = wav_transform(x)
                if feat_transform is not None:
                    h = feat_transform(h)

                # remove the last and first sample for training
                batch_x = x[:-1].unsqueeze(0)
                batch_h = h[:-1].transpose(0, 1).unsqueeze(0)
                batch_t = x[1:]

                yield (batch_x, batch_h), batch_t

            # use utterance batch with upsampling
            else:
                # remove last frame
                h = h[:-1]
                x = x[:-upsampling_factor+1]

                # perform pre-processing
                if wav_transform is not None:
                    x = wav_transform(x)
                if feat_transform is not None:
                    h = feat_transform(h)

                # remove the last and first sample for training
                batch_h = h.transpose(0, 1).unsqueeze(0)
                batch_x = x[:-1].unsqueeze(0)
                batch_t = x[1:]

                yield (batch_x, batch_h), batch_t

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
    model.cpu()
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iterations": iterations}
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    torch.save(checkpoint, checkpoint_dir + "/checkpoint-%d.pkl" % iterations)
    model.cuda()
    logging.info("%d-iter checkpoint created." % iterations)


def main():
    parser = argparse.ArgumentParser()
    # path setting
    parser.add_argument("--waveforms", required=True,
                        type=str, help="directory or list of wav files")
    parser.add_argument("--feats", required=True,
                        type=str, help="directory or list of aux feat files")
    parser.add_argument("--stats", required=True,
                        type=str, help="hdf5 file including statistics")
    parser.add_argument("--expdir", required=True,
                        type=str, help="directory to save the model")
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
    parser.add_argument("--dilation_repeat", default=1,
                        type=int, help="number of repeating of dilation")
    parser.add_argument("--kernel_size", default=2,
                        type=int, help="kernel size of dilated causal convolution")
    parser.add_argument("--upsampling_factor", default=0,
                        type=int, help="upsampling factor of aux features"
                                       "(if set 0, do not apply)")
    parser.add_argument("--use_speaker_code", default=False,
                        type=strtobool, help="flag to use speaker code")
    # network training setting
    parser.add_argument("--lr", default=1e-4,
                        type=float, help="learning rate")
    parser.add_argument("--weight_decay", default=0.0,
                        type=float, help="weight decay coefficient")
    parser.add_argument("--batch_size", default=20000,
                        type=int, help="batch size (if set 0, utterance batch will be used)")
    parser.add_argument("--iters", default=200000,
                        type=int, help="number of iterations")
    # other setting
    parser.add_argument("--checkpoints", default=10000,
                        type=int, help="how frequent saving model")
    parser.add_argument("--intervals", default=100,
                        type=int, help="log interval")
    parser.add_argument("--seed", default=1,
                        type=int, help="seed number")
    parser.add_argument("--resume", default=None,
                        type=str, help="model path to restart training")
    parser.add_argument("--verbose", default=1,
                        type=int, help="log level")
    args = parser.parse_args()

    # make experimental directory
    if not os.path.exists(args.expdir):
        os.makedirs(args.expdir)

    # set log level
    if args.verbose == 1:
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S',
                            filename=args.expdir + "/train.log")
        logging.getLogger().addHandler(logging.StreamHandler())
    elif args.verbose > 1:
        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S',
                            filename=args.expdir + "/train.log")
        logging.getLogger().addHandler(logging.StreamHandler())
    else:
        logging.basicConfig(level=logging.WARN,
                            format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S',
                            filename=args.expdir + "/train.log")
        logging.getLogger().addHandler(logging.StreamHandler())
        logging.warn("logging is disabled.")

    # fix seed
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # save args as conf
    torch.save(args, args.expdir + "/model.conf")

    # # define network
    model = WaveNet(
        n_quantize=args.n_quantize,
        n_aux=args.n_aux,
        n_resch=args.n_resch,
        n_skipch=args.n_skipch,
        dilation_depth=args.dilation_depth,
        dilation_repeat=args.dilation_repeat,
        kernel_size=args.kernel_size,
        upsampling_factor=args.upsampling_factor)
    logging.info(model)
    model.apply(initialize)
    model.train()

    # define loss and optimizer
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

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
    if os.path.isdir(args.waveforms):
        filenames = sorted(find_files(args.waveforms, "*.wav", use_dir_name=False))
        wav_list = [args.waveforms + "/" + filename for filename in filenames]
        feat_list = [args.feats + "/" + filename.replace(".wav", ".h5") for filename in filenames]
    elif os.path.isfile(args.waveforms):
        wav_list = read_txt(args.waveforms)
        feat_list = read_txt(args.feats)
    else:
        logging.error("--waveforms should be directory or list.")
        sys.exit(1)
    assert len(wav_list) == len(feat_list)
    logging.info("number of training data = %d." % len(wav_list))
    generator = train_generator(
            wav_list, feat_list,
            receptive_field=model.receptive_field,
            batch_size=args.batch_size,
            wav_transform=wav_transform,
            feat_transform=feat_transform,
            shuffle=True,
            upsampling_factor=args.upsampling_factor,
            use_speaker_code=args.use_speaker_code)
    while not generator.queue.full():
        time.sleep(0.1)

    # resume
    if args.resume is not None:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        iterations = checkpoint["iterations"]
        logging.info("restored from %d-iter checkpoint." % iterations)
    else:
        iterations = 0

    # send to gpu
    if torch.cuda.is_available():
        model.cuda()
        criterion.cuda()
    else:
        logging.error("gpu is not available. please check the setting.")
        sys.exit(1)

    # train
    loss = 0
    total = 0
    for i in six.moves.range(iterations, args.iters):
        start = time.time()
        (batch_x, batch_h), batch_t = generator.next()
        batch_output = model(batch_x, batch_h)
        batch_loss = criterion(batch_output[model.receptive_field:],
                               batch_t[model.receptive_field:])
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        loss += batch_loss.data[0]
        total += time.time() - start
        logging.debug("batch loss = %.3f (%.3f sec / batch)" % (
            batch_loss.data[0], time.time()-start))

        # report progress
        if (i + 1) % args.intervals == 0:
            logging.info("(iter:%d) average loss = %.6f (%.3f sec / batch)" % (
                i + 1, loss / args.intervals, total / args.intervals))
            logging.info("estimated required time = "
                         "{0.days:02}:{0.hours:02}:{0.minutes:02}:{0.seconds:02}"
                         .format(relativedelta(
                             seconds=int((args.iters - (i + 1)) * (total / args.intervals)))))
            loss = 0
            total = 0

        # save intermidiate model
        if (i + 1) % args.checkpoints == 0:
            save_checkpoint(args.expdir, model, optimizer, i + 1)

    # save final model
    model.cpu()
    torch.save({"model": model.state_dict()}, args.expdir + "/checkpoint-final.pkl")
    logging.info("final checkpoint created.")


if __name__ == "__main__":
    main()
