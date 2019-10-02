# -*- coding: utf-8 -*-

# Copyright 2017 Tomoki Hayashi (Nagoya University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
import os
import shutil

import numpy as np
import pytest

from scipy.io import wavfile

from wavenet_vocoder.bin.calc_stats import calc_stats
from wavenet_vocoder.bin.feature_extract import melcepstrum_extract
from wavenet_vocoder.bin.feature_extract import melspectrogram_extract
from wavenet_vocoder.bin.feature_extract import world_feature_extract
from wavenet_vocoder.bin.noise_shaping import convert_mcep_to_mlsa_coef
from wavenet_vocoder.bin.noise_shaping import noise_shaping
from wavenet_vocoder.utils import check_hdf5
from wavenet_vocoder.utils import find_files
from wavenet_vocoder.utils import read_hdf5
from wavenet_vocoder.utils import write_hdf5


def make_dummy_wav(name, maxlen=32000, fs=16000):
    length = np.random.randint(maxlen // 2, maxlen)
    x = np.random.randn(length)
    x = x / np.abs(x).max()
    x = np.int16(x * (np.iinfo(np.int16).max + 1))
    wavfile.write(name, fs, x)


def make_args(**kwargs):
    defaults = dict(
        hdf5dir="tmp/hdf5",
        wavdir="tmp/wav_filtered",
        outdir="tmp/wav_nwf",
        stats="tmp/stats.h5",
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
        fmin=None,
        fmax=None,
        mag=0.5,
        save_wav=True,
        inv=False,
    )
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


@pytest.mark.parametrize("feature_type", [
    ("melspc"), ("world"), ("mcep"),
])
def test_preprocessing(feature_type):
    # make arguments
    args = make_args(feature_type=feature_type)

    # prepare dummy wav files
    wavdir = "tmp/wav"
    if not os.path.exists(wavdir):
        os.makedirs(wavdir)
    for i in range(5):
        make_dummy_wav(wavdir + "/%d.wav" % i, 8000, args.fs)

    # feature extract
    wav_list = find_files(wavdir, "*.wav")
    if not os.path.exists(args.wavdir):
        os.makedirs(args.wavdir)
    if args.feature_type == "world":
        world_feature_extract(wav_list, args)
    elif args.feature_type == "melspc":
        melspectrogram_extract(wav_list, args)
    else:
        melcepstrum_extract(wav_list, args)

    # calc_stats
    file_list = find_files(args.hdf5dir, "*.h5")
    calc_stats(file_list, args)

    # noise shaping
    if feature_type != "melspc":
        wav_list = find_files(args.wavdir, "*.wav")
        if not os.path.exists(args.outdir):
            os.makedirs(args.outdir)
        if not check_hdf5(args.stats, "/mlsa/coef"):
            avg_mcep = read_hdf5(args.stats, args.feature_type + "/mean")
            if args.feature_type == "world":
                avg_mcep = avg_mcep[args.mcep_dim_start:args.mcep_dim_end]
            mlsa_coef = convert_mcep_to_mlsa_coef(avg_mcep, args.mag, args.mcep_alpha)
            write_hdf5(args.stats, "/mlsa/coef", mlsa_coef)
            write_hdf5(args.stats, "/mlsa/alpha", args.mcep_alpha)
        noise_shaping(wav_list, args)

    # remove
    shutil.rmtree("tmp")
