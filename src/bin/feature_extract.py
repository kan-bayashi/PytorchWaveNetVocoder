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

import librosa
import numpy as np

from scipy.interpolate import interp1d
from scipy.io import wavfile
from scipy.signal import firwin
from scipy.signal import lfilter
from sprocket.speech.feature_extractor import FeatureExtractor

from utils import find_files
from utils import read_txt
from utils import write_hdf5


def low_cut_filter(x, fs, cutoff=70):
    """FUNCTION TO APPLY LOW CUT FILTER

    Args:
        x (ndarray): Waveform sequence
        fs (int): Sampling frequency
        cutoff (float): Cutoff frequency of low cut filter

    Return:
        (ndarray): Low cut filtered waveform sequence
    """

    nyquist = fs // 2
    norm_cutoff = cutoff / nyquist

    # low cut filter
    fil = firwin(255, norm_cutoff, pass_zero=False)
    lcf_x = lfilter(fil, 1, x)

    return lcf_x


def low_pass_filter(x, fs, cutoff=70, padding=True):
    """FUNCTION TO APPLY LOW PASS FILTER

    Args:
        x (ndarray): Waveform sequence
        fs (int): Sampling frequency
        cutoff (float): Cutoff frequency of low pass filter

    Return:
        (ndarray): Low pass filtered waveform sequence
    """

    nyquist = fs // 2
    norm_cutoff = cutoff / nyquist

    # low cut filter
    numtaps = 255
    fil = firwin(numtaps, norm_cutoff)
    x_pad = np.pad(x, (numtaps, numtaps), 'edge')
    lpf_x = lfilter(fil, 1, x_pad)
    lpf_x = lpf_x[numtaps + numtaps // 2: -numtaps // 2]

    return lpf_x


def convert_continuos_f0(f0):
    """CONVERT F0 TO CONTINUOUS F0

    Args:
        f0 (ndarray): original f0 sequence with the shape (T)

    Return:
        (ndarray): continuous f0 with the shape (T)
    """
    # get uv information as binary
    uv = np.float32(f0 != 0)

    # get start and end of f0
    if (f0 == 0).all():
        logging.warn("all of the f0 values are 0.")
        return uv, f0
    start_f0 = f0[f0 != 0][0]
    end_f0 = f0[f0 != 0][-1]

    # padding start and end of f0 sequence
    start_idx = np.where(f0 == start_f0)[0][0]
    end_idx = np.where(f0 == end_f0)[0][-1]
    f0[:start_idx] = start_f0
    f0[end_idx:] = end_f0

    # get non-zero frame index
    nz_frames = np.where(f0 != 0)[0]

    # perform linear interpolation
    f = interp1d(nz_frames, f0[nz_frames])
    cont_f0 = f(np.arange(0, f0.shape[0]))

    return uv, cont_f0


def world_feature_extract(wav_list, args):
    """EXTRACT WORLD FEATURE VECTOR"""
    # define feature extractor
    feature_extractor = FeatureExtractor(
        analyzer="world",
        fs=args.fs,
        shiftms=args.shiftms,
        minf0=args.minf0,
        maxf0=args.maxf0,
        fftl=args.fftl)

    for i, wav_name in enumerate(wav_list):
        logging.info("now processing %s (%d/%d)" % (wav_name, i + 1, len(wav_list)))
        # load wavfile and apply low cut filter
        fs, x = wavfile.read(wav_name)
        if x.dtype != np.int16:
            logging.warn("wav file format is not 16 bit PCM.")
        x = np.array(x, dtype=np.float32)
        if args.highpass_cutoff != 0:
            x = low_cut_filter(x, fs, cutoff=args.highpass_cutoff)

        # check sampling frequency
        if not fs == args.fs:
            logging.error("sampling frequency is not matched.")
            sys.exit(1)

        # extract features
        f0, _, _ = feature_extractor.analyze(x)
        uv, cont_f0 = convert_continuos_f0(f0)
        cont_f0_lpf = low_pass_filter(cont_f0, int(1.0 / (args.shiftms * 0.001)), cutoff=20)
        codeap = feature_extractor.codeap()
        mcep = feature_extractor.mcep(dim=args.mcep_dim, alpha=args.mcep_alpha)

        # concatenate
        cont_f0_lpf = np.expand_dims(cont_f0_lpf, axis=-1)
        uv = np.expand_dims(uv, axis=-1)
        feats = np.concatenate([uv, cont_f0_lpf, mcep, codeap], axis=1)

        # save to hdf5
        hdf5name = args.hdf5dir + "/" + os.path.basename(wav_name).replace(".wav", ".h5")
        write_hdf5(hdf5name, "/world", feats)

        # overwrite wav file
        if args.highpass_cutoff != 0:
            wavfile.write(args.wavdir + "/" + os.path.basename(wav_name), fs, np.int16(x))


def melspectrogram_extract(wav_list, args):
    """EXTRACT MEL SPECTROGRAM"""
    # define feature extractor
    for i, wav_name in enumerate(wav_list):
        logging.info("now processing %s (%d/%d)" % (wav_name, i + 1, len(wav_list)))
        # load wavfile and apply low cut filter
        fs, x = wavfile.read(wav_name)
        if x.dtype != np.int16:
            logging.warn("wav file format is not 16 bit PCM.")
        x = np.array(x, dtype=np.float32)
        if args.highpass_cutoff != 0:
            x = low_cut_filter(x, fs, cutoff=args.highpass_cutoff)

        # check sampling frequency
        if not fs == args.fs:
            logging.error("sampling frequency is not matched.")
            sys.exit(1)

        # extract features
        x_norm = x / (np.iinfo(np.int16).max + 1)
        shiftl = int(args.shiftms * fs * 0.001)
        mspc = librosa.feature.melspectrogram(
            x_norm, fs, n_fft=args.fftl, hop_length=shiftl, n_mels=args.mspc_dim)
        mspc = np.log10(mspc.T)

        # save to hdf5
        hdf5name = args.hdf5dir + "/" + os.path.basename(wav_name).replace(".wav", ".h5")
        write_hdf5(hdf5name, "/melspc", np.float32(mspc))

        # overwrite wav file
        if args.highpass_cutoff != 0:
            wavfile.write(args.wavdir + "/" + os.path.basename(wav_name), fs, np.int16(x))


def main():
    parser = argparse.ArgumentParser(
        description="making feature file argsurations.")

    parser.add_argument(
        "--waveforms", default=None,
        help="directory or list of filename of input wavfile")
    parser.add_argument(
        "--hdf5dir", default=None,
        help="directory to save hdf5")
    parser.add_argument(
        "--wavdir", default=None,
        help="directory to save of preprocessed wav file")
    parser.add_argument(
        "--fs", default=16000,
        type=int, help="Sampling frequency")
    parser.add_argument(
        "--shiftms", default=5,
        type=int, help="Frame shift in msec")
    parser.add_argument(
        "--feature_type", default="world", choices=["world", "melspc"],
        type=str, help="feature type")
    parser.add_argument(
        "--mspc_dim", default=80,
        type=int, help="Dimension of mel spectrogram")
    parser.add_argument(
        "--minf0", default=40,
        type=int, help="minimum f0")
    parser.add_argument(
        "--maxf0", default=400,
        type=int, help="maximum f0")
    parser.add_argument(
        "--mcep_dim", default=24,
        type=int, help="Dimension of mel cepstrum")
    parser.add_argument(
        "--mcep_alpha", default=0.41,
        type=float, help="Alpha of mel cepstrum")
    parser.add_argument(
        "--fftl", default=1024,
        type=int, help="FFT length")
    parser.add_argument(
        "--highpass_cutoff", default=70,
        type=int, help="Cut off frequency in lowpass filter")
    parser.add_argument(
        "--n_jobs", default=10,
        type=int, help="number of parallel jobs")
    parser.add_argument(
        "--verbose", default=1,
        type=int, help="log message level")

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
    if not os.path.exists(args.wavdir):
        os.makedirs(args.wavdir)
    if not os.path.exists(args.hdf5dir):
        os.makedirs(args.hdf5dir)

    # divie list
    file_lists = np.array_split(file_list, args.n_jobs)
    file_lists = [f_list.tolist() for f_list in file_lists]

    # multi processing
    processes = []
    if args.feature_type == "world":
        target_fn = world_feature_extract
    else:
        # TODO(kan-bayashi): implement feature extraction of mel spectrum
        raise NotImplementedError("currently, support only world.")
    for f in file_lists:
        p = mp.Process(target=target_fn, args=(f, args,))
        p.start()
        processes.append(p)

    # wait for all process
    for p in processes:
        p.join()


if __name__ == "__main__":
    main()
