#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2017 Tomoki Hayashi (Nagoya University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

from __future__ import division
from __future__ import print_function

import argparse
import os

import numpy as np
import soundfile as sf

from calc_stats import calc_stats
from feature_extract import feature_extract
from noise_shaping import noise_shaping
from utils import find_files


def make_dummy_wav(name, length, fs=16000):
    x = np.random.randn(length)
    x = x / np.abs(x).max()
    sf.write(name, x, fs, "PCM_16")


def make_args(**kwargs):
    defaults = dict(
        hdf5dir="data/hdf5",
        wavdir="data/wav_filtered",
        writedir="data/wav_ns",
        stats="data/stats.h5",
        fs=16000,
        shiftms=5,
        minf0=40,
        maxf0=400,
        mcep_dim=24,
        mcep_alpha=0.41,
        fftl=1024,
        highpass_cutoff=70,
        mcep_dim_start=2,
        mcep_dim_end=25,
        mag=0.5,
        inv=False
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
    feature_extract(wav_list, args)

    # calc_stats
    file_list = find_files(args.hdf5dir, "*.h5")
    calc_stats(file_list, args)

    # noise shaping
    wav_list = find_files(args.wavdir, "*.wav")
    if not os.path.exists(args.writedir):
        os.makedirs(args.writedir)
    noise_shaping(wav_list, args)

    # assert list length
    wav_ns_list = find_files(args.writedir, "*.wav")
    assert len(wav_list) == len(file_list)
    assert len(wav_list) == len(wav_ns_list)
