#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2017 Tomoki Hayashi (Nagoya University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

from __future__ import division
from __future__ import print_function

import argparse
import os

import numpy as np

from scipy.io import wavfile

from calc_stats import calc_stats
from feature_extract import melcepstrum_extract
from feature_extract import melspectrogram_extract
from feature_extract import world_feature_extract
from noise_shaping import melcepstrum_noise_shaping
from noise_shaping import world_noise_shaping
from utils import find_files


def make_dummy_wav(name, maxlen=32000, fs=16000):
    length = np.random.randint(maxlen // 2, maxlen)
    x = np.random.randn(length)
    x = x / np.abs(x).max()
    x = np.int16(x * (np.iinfo(np.int16).max + 1))
    wavfile.write(name, fs, x)


def make_args(**kwargs):
    defaults = dict(
        hdf5dir="data/hdf5",
        wavdir="data/wav_filtered",
        writedir="data/wav_ns",
        stats="data/stats.h5",
        feature_type="world",
        fs=16000,
        shiftms=5,
        minf0=40,
        maxf0=400,
        mspc_dim=80,
        mcep_dim=24,
        mcep_alpha=0.41,
        fftl=1024,
        highpass_cutoff=70,
        mcep_dim_start=2,
        mcep_dim_end=25,
        mag=0.5,
        save_wav=True,
        inv=False,
    )
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def test_preprocessing():
    # make arguments
    args = make_args()

    # prepare dummy wav files
    wavdir = "data/wav"
    if not os.path.exists(wavdir):
        os.makedirs(wavdir)
    for i in range(5):
        make_dummy_wav(wavdir + "/%d.wav" % i, 8000, args.fs)

    # feature extract
    wav_list = find_files(wavdir, "*.wav")
    if not os.path.exists(args.wavdir):
        os.makedirs(args.wavdir)
    args.feature_type = "world"
    world_feature_extract(wav_list, args)
    args.feature_type = "melspc"
    melspectrogram_extract(wav_list, args)
    args.feature_type = "mcep"
    melcepstrum_extract(wav_list, args)

    # calc_stats
    file_list = find_files(args.hdf5dir, "*.h5")
    args.feature_type = "world"
    calc_stats(file_list, args)
    args.feature_type = "melspc"
    calc_stats(file_list, args)
    args.feature_type = "mcep"
    calc_stats(file_list, args)

    # noise shaping
    wav_list = find_files(args.wavdir, "*.wav")
    args.feature_type = "world"
    args.writedir = "data/wav_ns/world"
    if not os.path.exists(args.writedir):
        os.makedirs(args.writedir)
    world_noise_shaping(wav_list, args)
    args.feature_type = "mcep"
    args.writedir = "data/wav_ns/mcep"
    if not os.path.exists(args.writedir):
        os.makedirs(args.writedir)
    melcepstrum_noise_shaping(wav_list, args)
