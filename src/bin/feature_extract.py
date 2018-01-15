#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2017 Tomoki Hayashi (Nagoya University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

from __future__ import division, print_function

import argparse
import multiprocessing as mp
import os
import sys

import numpy as np
from numpy.matlib import repmat
from scipy.interpolate import interp1d
from scipy.io import wavfile
from scipy.signal import firwin, lfilter
from sprocket.speech.feature_extractor import FeatureExtractor

from utils import find_files, read_txt, write_hdf5

FS = 22050
SHIFTMS = 5
MINF0 = 70
MAXF0 = 210
MCEP_DIM = 34
MCEP_ALPHA = 0.455
FFTL = 1024
HIGHPASS_CUTOFF = 70
OVERWRITE = True


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
    lpf_x = lpf_x[numtaps+numtaps//2:-numtaps//2]

    return lpf_x


def extend_time(feats, upsampling_factor):
    """FUNCTION TO EXTEND TIME RESOLUTION

    Args:
        feats (ndarray): feature vector with the shape (T x D)
        upsampling_factor (int): upsampling_factor

    Return:
        (ndarray): extend feats with the shape (upsampling_factor*T x D)
    """
    # get number
    n_frames = feats.shape[0]
    n_dims = feats.shape[1]

    # extend time
    feats_extended = np.zeros((n_frames * upsampling_factor, n_dims))
    for j in range(n_frames):
        start_idx = j * upsampling_factor
        end_idx = (j + 1) * upsampling_factor
        feats_extended[start_idx: end_idx] = repmat(feats[j, :], upsampling_factor, 1)

    return feats_extended


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
        "--fs", default=FS,
        type=int, help="Sampling frequency")
    parser.add_argument(
        "--shiftms", default=SHIFTMS,
        type=int, help="Frame shift in msec")
    parser.add_argument(
        "--minf0", default=MINF0,
        type=int, help="minimum f0")
    parser.add_argument(
        "--maxf0", default=MAXF0,
        type=int, help="maximum f0")
    parser.add_argument(
        "--mcep_dim", default=MCEP_DIM,
        type=int, help="Dimension of mel cepstrum")
    parser.add_argument(
        "--mcep_alpha", default=MCEP_ALPHA,
        type=float, help="Alpha of mel cepstrum")
    parser.add_argument(
        "--fftl", default=FFTL,
        type=int, help="FFT length")
    parser.add_argument(
        "--highpass_cutoff", default=HIGHPASS_CUTOFF,
        type=int, help="Cut off frequency in lowpass filter")
    parser.add_argument(
        "--n_jobs", default=10,
        type=int, help="number of parallel jobs")
    parser.add_argument(
        "--verbose", default=1,
        type=int, help="log message level")

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
            minf0=args.minf0,
            maxf0=args.maxf0,
            fftl=args.fftl)

    # check directory existence
    if not os.path.exists(args.wavdir):
        os.makedirs(args.wavdir)
    if not os.path.exists(args.hdf5dir):
        os.makedirs(args.hdf5dir)

    def feature_extract(wav_list):
        for wav_name in wav_list:
            # load wavfile and apply low cut filter
            fs, x = wavfile.read(wav_name)
            x = np.array(x, dtype=np.float32)
            if args.highpass_cutoff != 0:
                x = low_cut_filter(x, fs, cutoff=args.highpass_cutoff)

            # check sampling frequency
            if not fs == args.fs:
                print("ERROR: sampling frequency is not matched.")
                sys.exit(1)

            # extract features
            f0, spc, ap = feature_extractor.analyze(x)
            uv, cont_f0 = convert_continuos_f0(f0)
            cont_f0_lpf = low_pass_filter(cont_f0, int(1.0/(args.shiftms*0.001)), cutoff=20)
            codeap = feature_extractor.codeap()
            mcep = feature_extractor.mcep(dim=args.mcep_dim, alpha=args.mcep_alpha)

            # concatenate
            cont_f0_lpf = np.expand_dims(cont_f0_lpf, axis=-1)
            uv = np.expand_dims(uv, axis=-1)
            feats = np.concatenate([uv, cont_f0_lpf, mcep, codeap], axis=1)

            # extend time resolution
            upsampling_factor = int(args.shiftms * fs * 0.001)
            feats_extended = extend_time(feats, upsampling_factor)

            # save to hdf5
            feats_extended = feats_extended.astype(np.float32)
            hdf5name = args.hdf5dir + "/" + os.path.basename(wav_name).replace(".wav", ".h5")
            write_hdf5(hdf5name, "/feat_org", feats)
            write_hdf5(hdf5name, "/feat", feats_extended)

            # overwrite wav file
            if args.highpass_cutoff != 0:
                wavfile.write(args.wavdir + "/" + os.path.basename(wav_name), fs, np.int16(x))

    # divie list
    file_lists = np.array_split(file_list, args.n_jobs)
    file_lists = [f_list.tolist() for f_list in file_lists]

    # multi processing
    processes = []
    for f in file_lists:
        p = mp.Process(target=feature_extract, args=(f,))
        p.start()
        processes.append(p)

    # wait for all process
    for p in processes:
        p.join()


if __name__ == "__main__":
    main()
