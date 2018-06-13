#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2017 Tomoki Hayashi (Nagoya University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

from __future__ import division

import argparse
import logging
import multiprocessing as mp
import os
import sys

from distutils.util import strtobool

import numpy as np

from scipy.io import wavfile
from sprocket.speech.synthesizer import Synthesizer

from feature_extract import low_cut_filter
from utils import find_files
from utils import read_hdf5
from utils import read_txt


def world_noise_shaping(wav_list, args):
    """APPLY NOISE SHAPING USING WORLD MCEP"""
    # define synthesizer
    synthesizer = Synthesizer(
        fs=args.fs,
        shiftms=args.shiftms,
        fftl=args.fftl)

    for i, wav_name in enumerate(wav_list):
        logging.info("now processing %s (%d/%d)" % (wav_name, i + 1, len(wav_list)))

        # load wavfile and apply low cut filter
        fs, x = wavfile.read(wav_name)
        if x.dtype != np.int16:
            logging.warn("wav file format is not 16 bit PCM.")
        x = np.float64(x)

        # check sampling frequency
        if not fs == args.fs:
            logging.error("sampling frequency is not matched.")
            sys.exit(1)

        # get frame number
        num_frames = int(1000 * len(x) / fs / args.shiftms) + 1

        # load average mcep
        mlsa_coef = read_hdf5(args.stats, "/world/mean")
        mlsa_coef = mlsa_coef[args.mcep_dim_start:args.mcep_dim_end] * args.mag
        mlsa_coef[0] = 0.0
        if args.inv:
            mlsa_coef[1:] = -1.0 * mlsa_coef[1:]
        mlsa_coef = np.float64(np.tile(mlsa_coef, [num_frames, 1]))

        # synthesis and write
        x_ns = synthesizer.synthesis_diff(
            x, mlsa_coef, alpha=args.mcep_alpha)
        x_ns = low_cut_filter(x_ns, args.fs, cutoff=70)
        write_name = args.writedir + "/" + os.path.basename(wav_name)
        wavfile.write(write_name, args.fs, np.int16(x_ns))


def melcepstrum_noise_shaping(wav_list, args):
    """APPLY NOISE SHAPING USING STFT-BASED MCEP"""
    # define synthesizer
    synthesizer = Synthesizer(
        fs=args.fs,
        shiftms=args.shiftms,
        fftl=args.fftl)

    for i, wav_name in enumerate(wav_list):
        logging.info("now processing %s (%d/%d)" % (wav_name, i + 1, len(wav_list)))

        # load wavfile and apply low cut filter
        fs, x = wavfile.read(wav_name)
        if x.dtype != np.int16:
            logging.warn("wav file format is not 16 bit PCM.")
        x = np.float64(x)

        # check sampling frequency
        if not fs == args.fs:
            logging.error("sampling frequency is not matched.")
            sys.exit(1)

        # get frame number
        num_frames = int(1000 * len(x) / fs / args.shiftms) + 1

        # load average mcep
        mlsa_coef = read_hdf5(args.stats, "/mcep/mean") * args.mag
        mlsa_coef[0] = 0.0
        if args.inv:
            mlsa_coef[1:] = -1.0 * mlsa_coef[1:]
        mlsa_coef = np.float64(np.tile(mlsa_coef, [num_frames, 1]))

        # synthesis and write
        x_ns = synthesizer.synthesis_diff(
            x, mlsa_coef, alpha=args.mcep_alpha)
        x_ns = low_cut_filter(x_ns, args.fs, cutoff=70)
        write_name = args.writedir + "/" + os.path.basename(wav_name)
        wavfile.write(write_name, args.fs, np.int16(x_ns))


def main():
    parser = argparse.ArgumentParser(
        description="making feature file argsurations.")

    parser.add_argument(
        "--waveforms", default=None,
        help="directory or list of filename of input wavfile")
    parser.add_argument(
        "--stats", default=None,
        help="filename of hdf5 format")
    parser.add_argument(
        "--writedir", default=None,
        help="directory to save preprocessed wav file")
    parser.add_argument(
        "--fs", default=16000,
        type=int, help="Sampling frequency")
    parser.add_argument(
        "--shiftms", default=5,
        type=float, help="Frame shift in msec")
    parser.add_argument(
        "--fftl", default=1024,
        type=int, help="FFT length")
    parser.add_argument(
        "--feature_type", default="world", choices=["world", "mcep", "melspc"],
        type=str, help="feature type")
    parser.add_argument(
        "--mcep_dim_start", default=2,
        type=int, help="Start index of mel cepstrum")
    parser.add_argument(
        "--mcep_dim_end", default=27,
        type=int, help="End index of mel cepstrum")
    parser.add_argument(
        "--mcep_alpha", default=0.41,
        type=float, help="Alpha of mel cepstrum")
    parser.add_argument(
        "--mag", default=0.5,
        type=float, help="magnification of noise shaping")
    parser.add_argument(
        "--verbose", default=1,
        type=int, help="log message level")
    parser.add_argument(
        '--n_jobs', default=10,
        type=int, help="number of parallel jobs")
    parser.add_argument(
        '--inv', default=False, type=strtobool,
        help="if True, inverse filtering will be performed")

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

    # read list
    if os.path.isdir(args.waveforms):
        file_list = sorted(find_files(args.waveforms, "*.wav"))
    else:
        file_list = read_txt(args.waveforms)
    logging.info("number of utterances = %d" % len(file_list))

    # check directory existence
    if not os.path.exists(args.writedir):
        os.makedirs(args.writedir)

    # divie list
    file_lists = np.array_split(file_list, args.n_jobs)
    file_lists = [f_list.tolist() for f_list in file_lists]

    # multi processing
    processes = []
    if args.feature_type == "world":
        target_fn = world_noise_shaping
    elif args.feature_type == "mcep":
        target_fn = melcepstrum_noise_shaping
    else:
        # TODO(kan-bayashi): implement noise shaping using melspectrogram
        NotImplementedError("currently, support only world and mcep.")
    for f in file_lists:
        p = mp.Process(target=target_fn, args=(f, args,))
        p.start()
        processes.append(p)

    # wait for all process
    for p in processes:
        p.join()


if __name__ == "__main__":
    main()
