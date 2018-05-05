#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2017 Tomoki Hayashi (Nagoya University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

from __future__ import division

import argparse
import logging
import math
import os
import sys

import numpy as np
import soundfile as sf
import torch
import torch.multiprocessing as mp

from sklearn.preprocessing import StandardScaler
from torchvision import transforms

from utils import extend_time
from utils import find_files
from utils import read_hdf5
from utils import read_txt
from utils import shape_hdf5
from wavenet import decode_mu_law
from wavenet import encode_mu_law
from wavenet import WaveNet


def pad_list(batch_list, pad_value=0.0):
    """FUNCTION TO PAD VALUE

    Args:
        batch_list (list): list of batch, where the shape of i-th batch (T_i, C)
        pad_value (float): value to pad

    Return:
        (ndarray): padded batch with the shape (B, T_max, C)

    """
    batch_size = len(batch_list)
    maxlen = max([batch.shape[0] for batch in batch_list])
    n_feats = batch_list[0].shape[-1]
    batch_pad = np.zeros((batch_size, maxlen, n_feats))
    for idx, batch in enumerate(batch_list):
        batch_pad[idx, :batch.shape[0]] = batch

    return batch_pad


def decode_generator(feat_list,
                     batch_size=32,
                     feature_type="world",
                     wav_transform=None,
                     feat_transform=None,
                     upsampling_factor=80,
                     use_upsampling_layer=True,
                     use_speaker_code=False):
    """DECODE BATCH GENERATOR

    Args:
        featdir (str): directory including feat files
        batch_size (int): batch size in decoding
        feature_type (str): feature type
        wav_transform (func): preprocessing function for waveform
        feat_transform (func): preprocessing function for aux feats
        upsampling_factor (int): upsampling factor
        use_upsampling_layer (bool): whether to use upsampling layer
        use_speaker_code (bool): whether to use speaker code

    Return:
        (object): generator instance
    """
    # ---------------------------
    # sample-by-sample generation
    # ---------------------------
    if batch_size == 1:
        for featfile in feat_list:
            x = np.zeros((1))
            h = read_hdf5(featfile, "/" + feature_type)
            if not use_upsampling_layer:
                h = extend_time(h, upsampling_factor)
            if use_speaker_code:
                sc = read_hdf5(featfile, "/speaker_code")
                sc = np.tile(sc, [h.shape[0], 1])
                h = np.concatenate([h, sc], axis=1)

            # perform pre-processing
            if wav_transform is not None:
                x = wav_transform(x)
            if feat_transform is not None:
                h = feat_transform(h)

            # convert to torch variable
            x = torch.from_numpy(x).long()
            h = torch.from_numpy(h).float()
            x = x.unsqueeze(0)  # 1 => 1 x 1
            h = h.transpose(0, 1).unsqueeze(0)  # T x C => 1 x C x T

            # send to cuda
            if torch.cuda.is_available():
                x = x.cuda()
                h = h.cuda()

            # get target length and file id
            if not use_upsampling_layer:
                n_samples = h.size(2) - 1
            else:
                n_samples = h.size(2) * upsampling_factor - 1
            feat_id = os.path.basename(featfile).replace(".h5", "")

            yield feat_id, (x, h, n_samples)

    # ----------------
    # batch generation
    # ----------------
    else:
        # sort with the feature length
        shape_list = [shape_hdf5(f, "/" + feature_type)[0] for f in feat_list]
        idx = np.argsort(shape_list)
        feat_list = [feat_list[i] for i in idx]

        # divide into batch list
        n_batch = math.ceil(len(feat_list) / batch_size)
        batch_lists = np.array_split(feat_list, n_batch)
        batch_lists = [f.tolist() for f in batch_lists]

        for batch_list in batch_lists:
            batch_x = []
            batch_h = []
            n_samples_list = []
            feat_ids = []
            for featfile in batch_list:
                # make seed waveform and load aux feature
                x = np.zeros((1))
                h = read_hdf5(featfile, "/" + feature_type)
                if not use_upsampling_layer:
                    h = extend_time(h, upsampling_factor)
                if use_speaker_code:
                    sc = read_hdf5(featfile, "/speaker_code")
                    sc = np.tile(sc, [h.shape[0], 1])
                    h = np.concatenate([h, sc], axis=1)

                # perform pre-processing
                if wav_transform is not None:
                    x = wav_transform(x)
                if feat_transform is not None:
                    h = feat_transform(h)

                # append to list
                batch_x += [x]
                batch_h += [h]
                if not use_upsampling_layer:
                    n_samples_list += [h.shape[0] - 1]
                else:
                    n_samples_list += [h.shape[0] * upsampling_factor - 1]
                feat_ids += [os.path.basename(featfile).replace(".h5", "")]

            # convert list to ndarray
            batch_x = np.stack(batch_x, axis=0)
            batch_h = pad_list(batch_h)

            # convert to torch variable
            batch_x = torch.from_numpy(batch_x).long()
            batch_h = torch.from_numpy(batch_h).float().transpose(1, 2)

            # send to cuda
            if torch.cuda.is_available():
                batch_x = batch_x.cuda()
                batch_h = batch_h.cuda()

            yield feat_ids, (batch_x, batch_h, n_samples_list)


def main():
    parser = argparse.ArgumentParser()
    # decode setting
    parser.add_argument("--feats", required=True,
                        type=str, help="list or directory of aux feat files")
    parser.add_argument("--stats", required=True,
                        type=str, help="hdf5 file including statistics")
    parser.add_argument("--checkpoint", required=True,
                        type=str, help="model file")
    parser.add_argument("--config", required=True,
                        type=str, help="configure file")
    parser.add_argument("--outdir", required=True,
                        type=str, help="directory to save generated samples")
    parser.add_argument("--fs", default=16000,
                        type=int, help="sampling rate")
    parser.add_argument("--batch_size", default=32,
                        type=int, help="number of batch size in decoding")
    parser.add_argument("--n_gpus", default=1,
                        type=int, help="number of gpus")
    # other setting
    parser.add_argument("--intervals", default=1000,
                        type=int, help="log interval")
    parser.add_argument("--seed", default=1,
                        type=int, help="seed number")
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

    # show argmument
    for key, value in vars(args).items():
        logging.info("%s = %s" % (key, str(value)))

    # check directory existence
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    # fix seed
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # load config
    config = torch.load(args.config)

    # get file list
    if os.path.isdir(args.feats):
        feat_list = sorted(find_files(args.feats, "*.h5"))
    elif os.path.isfile(args.feats):
        feat_list = read_txt(args.feats)
    else:
        logging.error("--feats should be directory or list.")
        sys.exit(1)

    # prepare the file list for parallel decoding
    feat_lists = np.array_split(feat_list, args.n_gpus)
    feat_lists = [f_list.tolist() for f_list in feat_lists]

    # define transform
    scaler = StandardScaler()
    scaler.mean_ = read_hdf5(args.stats, "/" + config.feature_type + "/mean")
    scaler.scale_ = read_hdf5(args.stats, "/" + config.feature_type + "/scale")
    wav_transform = transforms.Compose([
        lambda x: encode_mu_law(x, config.n_quantize)])
    feat_transform = transforms.Compose([
        lambda x: scaler.transform(x)])

    # define gpu decode function
    def gpu_decode(feat_list, gpu):
        # set default gpu and do not track gradient
        torch.cuda.set_device(gpu)
        torch.set_grad_enabled(False)

        # define model and load parameters
        if config.use_upsampling_layer:
            upsampling_factor = config.upsampling_factor
        else:
            upsampling_factor = 0
        model = WaveNet(
            n_quantize=config.n_quantize,
            n_aux=config.n_aux,
            n_resch=config.n_resch,
            n_skipch=config.n_skipch,
            dilation_depth=config.dilation_depth,
            dilation_repeat=config.dilation_repeat,
            kernel_size=config.kernel_size,
            upsampling_factor=upsampling_factor)
        model.load_state_dict(torch.load(
            args.checkpoint,
            map_location=lambda storage,
            loc: storage)["model"])
        model.eval()
        model.cuda()

        # define generator
        generator = decode_generator(
            feat_list,
            batch_size=args.batch_size,
            feature_type=config.feature_type,
            wav_transform=wav_transform,
            feat_transform=feat_transform,
            upsampling_factor=config.upsampling_factor,
            use_upsampling_layer=config.use_upsampling_layer,
            use_speaker_code=config.use_speaker_code)

        # decode
        if args.batch_size > 1:
            for feat_ids, (batch_x, batch_h, n_samples_list) in generator:
                logging.info("decoding start")
                samples_list = model.batch_fast_generate(
                    batch_x, batch_h, n_samples_list, args.intervals)
                for feat_id, samples in zip(feat_ids, samples_list):
                    wav = decode_mu_law(samples, config.n_quantize)
                    sf.write(args.outdir + "/" + feat_id + ".wav", wav, args.fs, "PCM_16")
                    logging.info("wrote %s.wav in %s." % (feat_id, args.outdir))
        else:
            for feat_id, (x, h, n_samples) in generator:
                logging.info("decoding %s (length = %d)" % (feat_id, n_samples))
                samples = model.fast_generate(x, h, n_samples, args.intervals)
                wav = decode_mu_law(samples, config.n_quantize)
                sf.write(args.outdir + "/" + feat_id + ".wav", wav, args.fs, "PCM_16")
                logging.info("wrote %s.wav in %s." % (feat_id, args.outdir))

    # parallel decode
    processes = []
    for gpu, feat_list in enumerate(feat_lists):
        p = mp.Process(target=gpu_decode, args=(feat_list, gpu,))
        p.start()
        processes.append(p)

    # wait for all process
    for p in processes:
        p.join()


if __name__ == "__main__":
    main()
