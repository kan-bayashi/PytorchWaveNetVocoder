#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2017 Tomoki Hayashi (Nagoya University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

from __future__ import division

import argparse
import logging
import os
import sys
import time

from dateutil.relativedelta import relativedelta
from distutils.util import strtobool

import numpy as np
import six
import soundfile as sf
import torch

from sklearn.preprocessing import StandardScaler
from torch import nn
from torchvision import transforms

from utils import background
from utils import extend_time
from utils import find_files
from utils import read_hdf5
from utils import read_txt
from wavenet import encode_mu_law
from wavenet import initialize
from wavenet import WaveNet


def validate_length(x, y, upsampling_factor=None):
    """FUNCTION TO VALIDATE LENGTH

    Args:
        x (ndarray): numpy.ndarray with x.shape[0] = len_x
        y (ndarray): numpy.ndarray with y.shape[0] = len_y
        upsampling_factor (int): upsampling factor

    Returns:
        (ndarray): length adjusted x with same length y
        (ndarray): length adjusted y with same length x
    """
    if upsampling_factor is None:
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
def train_generator(wav_list, feat_list, receptive_field,
                    batch_length=None,
                    batch_size=1,
                    feature_type="world",
                    wav_transform=None,
                    feat_transform=None,
                    shuffle=True,
                    upsampling_factor=80,
                    use_upsampling_layer=True,
                    use_speaker_code=False):
    """TRAINING BATCH GENERATOR

    Args:
        wav_list (str): list of wav files
        feat_list (str): list of feat files
        receptive_field (int): size of receptive filed
        batch_length (int): batch length (if set None, utterance batch will be used.)
        batch_size (int): batch size (if batch_length = None, batch_size will be 1.)
        feature_type (str): auxiliary feature type
        wav_transform (func): preprocessing function for waveform
        feat_transform (func): preprocessing function for aux feats
        shuffle (bool): whether to shuffle the file list
        upsampling_factor (int): upsampling factor
        use_upsampling_layer (bool): whether to use upsampling layer
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

    # check batch_length
    if batch_length is not None and use_upsampling_layer:
        batch_mod = (receptive_field + batch_length) % upsampling_factor
        logging.warn("batch length is decreased due to upsampling (%d -> %d)" % (
            batch_length, batch_length - batch_mod))
        batch_length -= batch_mod

    # show warning
    if batch_length is None and batch_size > 1:
        logging.warn("in utterance batch mode, batchsize will be 1.")

    while True:
        batch_x, batch_h, batch_t = [], [], []
        # process over all of files
        for wavfile, featfile in zip(wav_list, feat_list):
            # load wavefrom and aux feature
            x, fs = sf.read(wavfile, dtype=np.float32)
            h = read_hdf5(featfile, "/" + feature_type)
            if not use_upsampling_layer:
                h = extend_time(h, upsampling_factor)
            if use_speaker_code:
                sc = read_hdf5(featfile, "/speaker_code")
                sc = np.tile(sc, [h.shape[0], 1])
                h = np.concatenate([h, sc], axis=1)

            # check both lengths are same
            logging.debug("before x length = %d" % x.shape[0])
            logging.debug("before h length = %d" % h.shape[0])
            if use_upsampling_layer:
                x, h = validate_length(x, h, upsampling_factor)
            else:
                x, h = validate_length(x, h)
            logging.debug("after x length = %d" % x.shape[0])
            logging.debug("after h length = %d" % h.shape[0])

            # ---------------------------------------
            # use mini batch without upsampling layer
            # ---------------------------------------
            if batch_length is not None and not use_upsampling_layer:
                # make buffer array
                if "x_buffer" not in locals():
                    x_buffer = np.empty((0), dtype=np.float32)
                    h_buffer = np.empty((0, h.shape[1]), dtype=np.float32)
                x_buffer = np.concatenate([x_buffer, x], axis=0)
                h_buffer = np.concatenate([h_buffer, h], axis=0)

                while len(x_buffer) > receptive_field + batch_length:
                    # get pieces
                    x_ = x_buffer[:receptive_field + batch_length]
                    h_ = h_buffer[:receptive_field + batch_length]

                    # perform pre-processing
                    if wav_transform is not None:
                        x_ = wav_transform(x_)
                    if feat_transform is not None:
                        h_ = feat_transform(h_)

                    # convert to torch variable
                    x_ = torch.from_numpy(x_).long()
                    h_ = torch.from_numpy(h_).float()

                    # remove the last and first sample for training
                    batch_x += [x_[:-1]]  # (T)
                    batch_h += [h_[:-1].transpose(0, 1)]  # (D x T)
                    batch_t += [x_[1:]]  # (T)

                    # update buffer
                    x_buffer = x_buffer[batch_length:]
                    h_buffer = h_buffer[batch_length:]

                    # return mini batch
                    if len(batch_x) == batch_size:
                        batch_x = torch.stack(batch_x)
                        batch_h = torch.stack(batch_h)
                        batch_t = torch.stack(batch_t)

                        # send to cuda
                        if torch.cuda.is_available():
                            batch_x = batch_x.cuda()
                            batch_h = batch_h.cuda()
                            batch_t = batch_t.cuda()

                        yield (batch_x, batch_h), batch_t

                        batch_x, batch_h, batch_t = [], [], []

            # ------------------------------------
            # use mini batch with upsampling layer
            # ------------------------------------
            elif batch_length is not None and use_upsampling_layer:
                # make buffer array
                if "x_buffer" not in locals():
                    x_buffer = np.empty((0), dtype=np.float32)
                    h_buffer = np.empty((0, h.shape[1]), dtype=np.float32)
                x_buffer = np.concatenate([x_buffer, x], axis=0)
                h_buffer = np.concatenate([h_buffer, h], axis=0)

                while len(h_buffer) > (receptive_field + batch_length) // upsampling_factor:
                    # set batch size
                    h_bs = (receptive_field + batch_length) // upsampling_factor
                    x_bs = h_bs * upsampling_factor + 1

                    # get pieces
                    h_ = h_buffer[:h_bs]
                    x_ = x_buffer[:x_bs]

                    # perform pre-processing
                    if wav_transform is not None:
                        x_ = wav_transform(x_)
                    if feat_transform is not None:
                        h_ = feat_transform(h_)

                    # convert to torch variable
                    x_ = torch.from_numpy(x_).long()
                    h_ = torch.from_numpy(h_).float()

                    # remove the last and first sample for training
                    batch_h += [h_.transpose(0, 1)]  # (D x T)
                    batch_x += [x_[:-1]]  # (T)
                    batch_t += [x_[1:]]  # (T)

                    # set shift size
                    h_ss = batch_length // upsampling_factor
                    x_ss = h_ss * upsampling_factor

                    # update buffer
                    h_buffer = h_buffer[h_ss:]
                    x_buffer = x_buffer[x_ss:]

                    # return mini batch
                    if len(batch_x) == batch_size:
                        batch_x = torch.stack(batch_x)
                        batch_h = torch.stack(batch_h)
                        batch_t = torch.stack(batch_t)

                        # send to cuda
                        if torch.cuda.is_available():
                            batch_x = batch_x.cuda()
                            batch_h = batch_h.cuda()
                            batch_t = batch_t.cuda()

                        yield (batch_x, batch_h), batch_t

                        batch_x, batch_h, batch_t = [], [], []

            # --------------------------------------------
            # use utterance batch without upsampling layer
            # --------------------------------------------
            elif batch_length is None and not use_upsampling_layer:
                # perform pre-processing
                if wav_transform is not None:
                    x = wav_transform(x)
                if feat_transform is not None:
                    h = feat_transform(h)

                # convert to torch variable
                x = torch.from_numpy(x).long()
                h = torch.from_numpy(h).float()

                # remove the last and first sample for training
                batch_x = x[:-1].unsqueeze(0)  # (1 x T)
                batch_h = h[:-1].transpose(0, 1).unsqueeze(0)  # (1 x D x T)
                batch_t = x[1:].unsqueeze(0)  # (1 x T)

                # send to cuda
                if torch.cuda.is_available():
                    batch_x = batch_x.cuda()
                    batch_h = batch_h.cuda()
                    batch_t = batch_t.cuda()

                yield (batch_x, batch_h), batch_t

            # -----------------------------------------
            # use utterance batch with upsampling layer
            # -----------------------------------------
            else:
                # remove last frame
                h = h[:-1]
                x = x[:-upsampling_factor + 1]

                # perform pre-processing
                if wav_transform is not None:
                    x = wav_transform(x)
                if feat_transform is not None:
                    h = feat_transform(h)

                # convert to torch variable
                x = torch.from_numpy(x).long()
                h = torch.from_numpy(h).float()

                # remove the last and first sample for training
                batch_h = h.transpose(0, 1).unsqueeze(0)  # (1 x D x T')
                batch_x = x[:-1].unsqueeze(0)  # (1 x T)
                batch_t = x[1:].unsqueeze(0)  # (1 x T)

                # send to cuda
                if torch.cuda.is_available():
                    batch_x = batch_x.cuda()
                    batch_h = batch_h.cuda()
                    batch_t = batch_t.cuda()

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
    parser.add_argument("--waveforms", required=True,
                        type=str, help="directory or list of wav files")
    parser.add_argument("--feats", required=True,
                        type=str, help="directory or list of aux feat files")
    parser.add_argument("--stats", required=True,
                        type=str, help="hdf5 file including statistics")
    parser.add_argument("--expdir", required=True,
                        type=str, help="directory to save the model")
    parser.add_argument("--feature_type", default="world", choices=["world", "melspc"],
                        type=str, help="feature type")
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
    parser.add_argument("--upsampling_factor", default=80,
                        type=int, help="upsampling factor of aux features")
    parser.add_argument("--use_upsampling_layer", default=True,
                        type=strtobool, help="flag to use upsampling layer")
    parser.add_argument("--use_speaker_code", default=False,
                        type=strtobool, help="flag to use speaker code")
    # network training setting
    parser.add_argument("--lr", default=1e-4,
                        type=float, help="learning rate")
    parser.add_argument("--weight_decay", default=0.0,
                        type=float, help="weight decay coefficient")
    parser.add_argument("--batch_length", default=20000,
                        type=int, help="batch length (if set 0, utterance batch will be used)")
    parser.add_argument("--batch_size", default=1,
                        type=int, help="batch size (if use utterance batch, batch_size will be 1.")
    parser.add_argument("--iters", default=200000,
                        type=int, help="number of iterations")
    # other setting
    parser.add_argument("--checkpoints", default=10000,
                        type=int, help="how frequent saving model")
    parser.add_argument("--intervals", default=100,
                        type=int, help="log interval")
    parser.add_argument("--seed", default=1,
                        type=int, help="seed number")
    parser.add_argument("--resume", default=None, nargs="?",
                        type=str, help="model path to restart training")
    parser.add_argument("--n_gpus", default=1,
                        type=int, help="number of gpus")
    parser.add_argument("--verbose", default=1,
                        type=int, help="log level")
    args = parser.parse_args()

    # set log level
    if args.verbose == 1:
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

    # show argmument
    for key, value in vars(args).items():
        logging.info("%s = %s" % (key, str(value)))

    # make experimental directory
    if not os.path.exists(args.expdir):
        os.makedirs(args.expdir)

    # fix seed
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # save args as conf
    torch.save(args, args.expdir + "/model.conf")

    # define network
    if args.use_upsampling_layer:
        upsampling_factor = args.upsampling_factor
    else:
        upsampling_factor = 0
    model = WaveNet(
        n_quantize=args.n_quantize,
        n_aux=args.n_aux,
        n_resch=args.n_resch,
        n_skipch=args.n_skipch,
        dilation_depth=args.dilation_depth,
        dilation_repeat=args.dilation_repeat,
        kernel_size=args.kernel_size,
        upsampling_factor=upsampling_factor)
    logging.info(model)
    model.apply(initialize)
    model.train()

    if args.n_gpus > 1:
        device_ids = range(args.n_gpus)
        model = torch.nn.DataParallel(model, device_ids)
        model.receptive_field = model.module.receptive_field
        if args.n_gpus > args.batch_size:
            logging.warn("batch size is less than number of gpus.")

    # define optimizer and loss
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    # define transforms
    scaler = StandardScaler()
    scaler.mean_ = read_hdf5(args.stats, "/" + args.feature_type + "/mean")
    scaler.scale_ = read_hdf5(args.stats, "/" + args.feature_type + "/scale")
    wav_transform = transforms.Compose([
        lambda x: encode_mu_law(x, args.n_quantize)])
    feat_transform = transforms.Compose([
        lambda x: scaler.transform(x)])

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
        batch_length=args.batch_length,
        batch_size=args.batch_size,
        feature_type=args.feature_type,
        wav_transform=wav_transform,
        feat_transform=feat_transform,
        shuffle=True,
        upsampling_factor=args.upsampling_factor,
        use_upsampling_layer=args.use_upsampling_layer,
        use_speaker_code=args.use_speaker_code)

    # charge minibatch in queue
    while not generator.queue.full():
        time.sleep(0.1)

    # resume model and optimizer
    if args.resume is not None and len(args.resume) != 0:
        checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage)
        iterations = checkpoint["iterations"]
        if args.n_gpus > 1:
            model.module.load_state_dict(checkpoint["model"])
        else:
            model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        logging.info("restored from %d-iter checkpoint." % iterations)
    else:
        iterations = 0

    # check gpu and then send to gpu
    if torch.cuda.is_available():
        model.cuda()
        criterion.cuda()
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.cuda()
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
        batch_loss = criterion(
            batch_output[:, model.receptive_field:].contiguous().view(-1, args.n_quantize),
            batch_t[:, model.receptive_field:].contiguous().view(-1))
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        loss += batch_loss.item()
        total += time.time() - start
        logging.debug("batch loss = %.3f (%.3f sec / batch)" % (
            batch_loss.item(), time.time() - start))

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
            if args.n_gpus > 1:
                save_checkpoint(args.expdir, model.module, optimizer, i + 1)
            else:
                save_checkpoint(args.expdir, model, optimizer, i + 1)

    # save final model
    if args.n_gpus > 1:
        torch.save({"model": model.module.state_dict()}, args.expdir + "/checkpoint-final.pkl")
    else:
        torch.save({"model": model.state_dict()}, args.expdir + "/checkpoint-final.pkl")
    logging.info("final checkpoint created.")


if __name__ == "__main__":
    main()
