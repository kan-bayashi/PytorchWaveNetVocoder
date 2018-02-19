#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2017 Tomoki Hayashi (Nagoya University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

from __future__ import division
from __future__ import print_function

import argparse
import multiprocessing as mp
import os
import sys

from distutils.util import strtobool

import numpy as np

from scipy.io import wavfile
from sprocket.speech.feature_extractor import FeatureExtractor
from sprocket.speech.synthesizer import Synthesizer

from feature_extract import low_cut_filter
from utils import find_files
from utils import read_hdf5
from utils import read_txt

FS = 22050
SHIFTMS = 5
FFTL = 1024
MCEP_DIM_START = 2
MCEP_DIM_END = 37
MCEP_ALPHA = 0.455
MAG = 0.5


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
        "--fs", default=FS,
        type=int, help="Sampling frequency")
    parser.add_argument(
        "--shiftms", default=SHIFTMS,
        type=int, help="Frame shift in msec")
    parser.add_argument(
        "--fftl", default=FFTL,
        type=int, help="FFT length")
    parser.add_argument(
        "--mcep_dim_start", default=MCEP_DIM_START,
        type=int, help="Start index of mel cepstrum")
    parser.add_argument(
        "--mcep_dim_end", default=MCEP_DIM_END,
        type=int, help="End index of mel cepstrum")
    parser.add_argument(
        "--mcep_alpha", default=MCEP_ALPHA,
        type=float, help="Alpha of mel cepstrum")
    parser.add_argument(
        "--mag", default=MAG,
        type=float, help="magnification of noise shaping")
    parser.add_argument(
        "--verbose", default=1,
        type=int, help="log message level")
    parser.add_argument(
        '--n_jobs', default=1,
        type=int, help="number of parallel jobs")
    parser.add_argument(
        '--inv', default=False, type=strtobool,
        help="if True, inverse filtering will be performed")
    args = parser.parse_args()

    # read list
    if os.path.isdir(args.waveforms):
        file_list = sorted(find_files(args.waveforms, "*.wav"))
    else:
        file_list = read_txt(args.waveforms)

    # define feature extractor
    feature_extractor = FeatureExtractor(
        analyzer="world",
        fs=args.fs,
        shiftms=args.shiftms,
        fftl=args.fftl)

    # define synthesizer
    synthesizer = Synthesizer(
        fs=args.fs,
        shiftms=args.shiftms,
        fftl=args.fftl)

    # check directory existence
    if not os.path.exists(args.writedir):
        os.makedirs(args.writedir)

    def noise_shaping(wav_list):
        for wav_name in wav_list:
            # load wavfile and apply low cut filter
            fs, x = wavfile.read(wav_name)
            wav_type = x.dtype
            x = np.array(x, dtype=np.float64)

            # check sampling frequency
            if not fs == args.fs:
                print("ERROR: sampling frequency is not matched.")
                sys.exit(1)

            # extract features (only for get the number of frames)
            f0, _, _ = feature_extractor.analyze(x)
            num_frames = f0.shape[0]

            # load average mcep
            mlsa_coef = read_hdf5(args.stats, "/mean")
            mlsa_coef = mlsa_coef[args.mcep_dim_start:args.mcep_dim_end] * args.mag
            mlsa_coef[0] = 0.0
            if args.inv:
                mlsa_coef[1:] = -1.0 * mlsa_coef[1:]
            mlsa_coef = np.tile(mlsa_coef, [num_frames, 1])

            # synthesis and write
            x_ns = synthesizer.synthesis_diff(
                x, mlsa_coef, alpha=args.mcep_alpha)
            x_ns = low_cut_filter(x_ns, args.fs, cutoff=70)
            if wav_type == np.int16:
                write_name = args.writedir + "/" + os.path.basename(wav_name)
                wavfile.write(write_name, args.fs, np.int16(x_ns))
            else:
                wavfile.write(write_name, args.fs, x_ns)

    # divie list
    file_lists = np.array_split(file_list, args.n_jobs)
    file_lists = [f_list.tolist() for f_list in file_lists]

    # multi processing
    processes = []
    for f in file_lists:
        p = mp.Process(target=noise_shaping, args=(f,))
        p.start()
        processes.append(p)

    # wait for all process
    for p in processes:
        p.join()


if __name__ == "__main__":
    main()
