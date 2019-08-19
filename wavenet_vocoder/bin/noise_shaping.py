#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2017 Tomoki Hayashi (Nagoya University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
import logging
import multiprocessing as mp
import os
import sys

from distutils.util import strtobool

import numpy as np
import pysptk

from scipy.io import wavfile

from wavenet_vocoder.utils import check_hdf5
from wavenet_vocoder.utils import find_files
from wavenet_vocoder.utils import read_hdf5
from wavenet_vocoder.utils import read_txt
from wavenet_vocoder.utils import write_hdf5


def convert_mcep_to_mlsa_coef(avg_mcep, mag, alpha):
    """CONVERT AVERAGE MEL-CEPTSRUM TO MLSA FILTER COEFFICIENT.

    Args:
        avg_mcep (ndarray): Averaged Mel-cepstrum (D,).
        mag (float): Magnification of noise shaping.
        alpha (float): All pass constant value.

    Return:
        ndarray: MLSA filter coefficient (D,).

    """
    avg_mcep *= mag
    avg_mcep[0] = 0.0
    coef = pysptk.mc2b(avg_mcep.astype(np.float64), alpha)
    assert np.isfinite(coef).all()
    return coef


def noise_shaping(wav_list, args):
    """APPLY NOISE SHAPING BASED ON MLSA FILTER."""
    # load coefficient of filter
    if check_hdf5(args.stats, "/mlsa/coef"):
        mlsa_coef = read_hdf5(args.stats, "/mlsa/coef")
        alpha = read_hdf5(args.stats, "/mlsa/alpha")
    else:
        raise KeyError("\"/mlsa/coef\" is not found in %s." % (args.stats))
    if args.inv:
        mlsa_coef *= -1.0

    # define synthesizer
    shiftl = int(args.fs / 1000 * args.shiftms)
    synthesizer = pysptk.synthesis.Synthesizer(
        pysptk.synthesis.MLSADF(
            order=mlsa_coef.shape[0] - 1,
            alpha=alpha),
        hopsize=shiftl
    )

    for i, wav_name in enumerate(wav_list):
        logging.info("now processing %s (%d/%d)" % (wav_name, i + 1, len(wav_list)))

        # load wavfile and apply low cut filter
        fs, x = wavfile.read(wav_name)
        if x.dtype != np.int16:
            logging.warning("wav file format is not 16 bit PCM.")
        x = np.float64(x)

        # check sampling frequency
        if not fs == args.fs:
            logging.error("sampling frequency is not matched.")
            sys.exit(1)

        # replicate coef for time-invariant filtering
        num_frames = int(len(x) / shiftl) + 1
        mlsa_coefs = np.float64(np.tile(mlsa_coef, [num_frames, 1]))

        # synthesis and write
        x_ns = synthesizer.synthesis(x, mlsa_coefs)
        write_name = args.outdir + "/" + os.path.basename(wav_name)
        wavfile.write(write_name, args.fs, np.int16(x_ns))


def main():
    """RUN NOISE SHAPING IN PARALLEL."""
    parser = argparse.ArgumentParser(
        description="making feature file argsurations.")

    parser.add_argument(
        "--waveforms", default=None,
        help="directory or list of filename of input wavfile")
    parser.add_argument(
        "--stats", default=None,
        help="filename of hdf5 format")
    parser.add_argument(
        "--outdir", default=None,
        help="directory to save preprocessed wav file")
    parser.add_argument(
        "--fs", default=16000,
        type=int, help="Sampling frequency")
    parser.add_argument(
        "--shiftms", default=5,
        type=float, help="Frame shift in msec")
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
        logging.basicConfig(level=logging.WARNING,
                            format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S')
        logging.warning("logging is disabled.")

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
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    # divie list
    file_lists = np.array_split(file_list, args.n_jobs)
    file_lists = [f_list.tolist() for f_list in file_lists]

    # calculate MLSA coef ans save it
    if not check_hdf5(args.stats, "/mlsa/coef"):
        avg_mcep = read_hdf5(args.stats, args.feature_type + "/mean")
        if args.feature_type == "world":
            avg_mcep = avg_mcep[args.mcep_dim_start:args.mcep_dim_end]
        mlsa_coef = convert_mcep_to_mlsa_coef(avg_mcep, args.mag, args.mcep_alpha)
        write_hdf5(args.stats, "/mlsa/coef", mlsa_coef)
        write_hdf5(args.stats, "/mlsa/alpha", args.mcep_alpha)

    # multi processing
    processes = []
    if args.feature_type == "melspc":
        # TODO(kan-bayashi): implement noise shaping using melspectrogram
        raise NotImplementedError("currently, support only world and mcep.")
    for f in file_lists:
        p = mp.Process(target=noise_shaping, args=(f, args,))
        p.start()
        processes.append(p)

    # wait for all process
    for p in processes:
        p.join()


if __name__ == "__main__":
    main()
