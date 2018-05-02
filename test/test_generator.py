#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2017 Tomoki Hayashi (Nagoya University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
import os

from decode import decode_generator
from feature_extract import melspectrogram_extract
from feature_extract import world_feature_extract
from train import train_generator
from utils import find_files

from test_preprocessing import make_args as make_feature_args
from test_preprocessing import make_dummy_wav


def make_train_generator_args(**kwargs):
    defaults = dict(
        wav_list=None,
        feat_list=None,
        receptive_field=1000,
        batch_length=3000,
        batch_size=5,
        feature_type="world",
        wav_transform=None,
        feat_transform=None,
        shuffle=False,
        upsampling_factor=80,
        use_upsampling_layer=True,
        use_speaker_code=False
    )
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def make_decode_generator_args(**kwargs):
    defaults = dict(
        feat_list=None,
        batch_size=5,
        feature_type="world",
        wav_transform=None,
        feat_transform=None,
        upsampling_factor=80,
        use_upsampling_layer=True,
        use_speaker_code=False
    )
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def test_train_generator():
    # make dummy wavfiles
    wavdir = "data/wav"
    if not os.path.exists(wavdir):
        os.makedirs(wavdir)
    for i in range(5):
        make_dummy_wav(wavdir + "/%d.wav" % i)

    # make features
    feat_args = make_feature_args()
    wav_list = find_files(wavdir, "*.wav")
    if not os.path.exists(feat_args.wavdir):
        os.makedirs(feat_args.wavdir)
    feat_args.feature_type = "melspc"
    melspectrogram_extract(wav_list, feat_args)
    feat_args.feature_type = "world"
    world_feature_extract(wav_list, feat_args)
    feat_list = find_files(feat_args.hdf5dir, "*.h5")

    for ft in ["world", "melspc"]:
        # ----------------------------------
        # minibatch without upsampling layer
        # ----------------------------------
        generator_args = make_train_generator_args(
            wav_list=wav_list,
            feat_list=feat_list,
            feature_type=ft,
            use_upsampling_layer=False,
            batch_length=10000,
            batch_size=5
        )
        generator = train_generator(**vars(generator_args))
        (x, h), t = next(generator)
        assert x.size(0) == t.size(0) == h.size(0)
        assert x.size(1) == t.size(1) == h.size(2)

        # ----------------------------------------
        # utterance batch without upsampling layer
        # ----------------------------------------
        generator_args = make_train_generator_args(
            wav_list=wav_list,
            feat_list=feat_list,
            feature_type=ft,
            use_upsampling_layer=False,
            batch_length=None,
            batch_size=5
        )
        generator = train_generator(**vars(generator_args))
        (x, h), t = next(generator)
        assert x.size(0) == t.size(0) == h.size(0) == 1
        assert x.size(1) == t.size(1) == h.size(2)

        # -------------------------------
        # minibatch with upsampling layer
        # -------------------------------
        generator_args = make_train_generator_args(
            wav_list=wav_list,
            feat_list=feat_list,
            feature_type=ft,
            use_upsampling_layer=True,
            batch_length=10000,
            batch_size=5
        )
        generator = train_generator(**vars(generator_args))
        (x, h), t = next(generator)
        assert x.size(0) == t.size(0) == h.size(0)
        assert x.size(1) == t.size(1) == h.size(2) * generator_args.upsampling_factor

        # -------------------------------------
        # utterance batch with upsampling layer
        # -------------------------------------
        generator_args = make_train_generator_args(
            wav_list=wav_list,
            feat_list=feat_list,
            feature_type=ft,
            use_upsampling_layer=True,
            batch_length=None,
            batch_size=5
        )
        generator = train_generator(**vars(generator_args))
        (x, h), t = next(generator)
        assert x.size(0) == t.size(0) == h.size(0) == 1
        assert x.size(1) == t.size(1) == h.size(2) * generator_args.upsampling_factor


def test_decode_generator():
    # make dummy wavfiles
    wavdir = "data/wav"
    if not os.path.exists(wavdir):
        os.makedirs(wavdir)
    for i in range(5):
        make_dummy_wav(wavdir + "/%d.wav" % i)

    # make features
    feat_args = make_feature_args()
    wav_list = find_files(wavdir, "*.wav")
    if not os.path.exists(feat_args.wavdir):
        os.makedirs(feat_args.wavdir)
    feat_args.feature_type = "melspc"
    melspectrogram_extract(wav_list, feat_args)
    feat_args.feature_type = "world"
    world_feature_extract(wav_list, feat_args)
    feat_list = find_files(feat_args.hdf5dir, "*.h5")

    for ft in ["world", "melspc"]:
        # ----------------------------------
        # non-batch without upsampling layer
        # ----------------------------------
        generator_args = make_decode_generator_args(
            feat_list=feat_list,
            feature_type=ft,
            use_upsampling_layer=False,
            batch_size=1
        )
        generator = decode_generator(**vars(generator_args))
        _, (x, h, n_samples) = next(generator)
        assert x.size(0) == h.size(0) == 1
        assert h.size(2) == n_samples + 1

        # -------------------------------
        # non-batch with upsampling layer
        # -------------------------------
        generator_args = make_decode_generator_args(
            feat_list=feat_list,
            feature_type=ft,
            use_upsampling_layer=True,
            batch_size=1
        )
        generator = decode_generator(**vars(generator_args))
        _, (x, h, n_samples) = next(generator)
        assert x.size(0) == h.size(0) == 1
        assert h.size(2) * generator_args.upsampling_factor == n_samples + 1

        # ----------------------------------
        # minibatch without upsampling layer
        # ----------------------------------
        generator_args = make_decode_generator_args(
            feat_list=feat_list,
            feature_type=ft,
            use_upsampling_layer=False,
            batch_size=5
        )
        generator = decode_generator(**vars(generator_args))
        _, (batch_x, batch_h, n_samples_list) = next(generator)
        assert batch_x.size(0) == batch_h.size(0) == len(n_samples_list)
        assert batch_h.size(2) == max(n_samples_list) + 1

        # -------------------------------
        # minibatch with upsampling layer
        # -------------------------------
        generator_args = make_decode_generator_args(
            feat_list=feat_list,
            feature_type=ft,
            use_upsampling_layer=True,
            batch_size=5
        )
        generator = decode_generator(**vars(generator_args))
        _, (batch_x, batch_h, n_samples_list) = next(generator)
        assert batch_x.size(0) == batch_h.size(0) == len(n_samples_list)
        assert batch_h.size(2) * generator_args.upsampling_factor == max(n_samples_list) + 1
