#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2017 Tomoki Hayashi (Nagoya University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
import logging

import numpy as np

from sklearn.preprocessing import StandardScaler

from utils import read_hdf5
from utils import read_txt
from utils import write_hdf5


def calc_stats(file_list, args):
    """CALCULATE STATISTICS"""
    scaler = StandardScaler()

    # process over all of data
    for i, filename in enumerate(file_list):
        logging.info("now processing %s (%d/%d)" % (filename, i + 1, len(file_list)))
        feat = read_hdf5(filename, "/" + args.feature_type)
        scaler.partial_fit(feat)

    # add uv term
    mean = scaler.mean_
    scale = scaler.scale_
    if args.feature_type == "world":
        mean[0] = 0.0
        scale[0] = 1.0

    # write to hdf5
    write_hdf5(args.stats, "/" + args.feature_type + "/mean", np.float32(mean))
    write_hdf5(args.stats, "/" + args.feature_type + "/scale", np.float32(scale))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--feats", default=None, required=True,
        type=str, help="name of the list of hdf5 files")
    parser.add_argument(
        "--stats", default=None, required=True,
        type=str, help="filename of hdf5 format")
    parser.add_argument(
        "--feature_type", default="world", choices=["world", "melspc", "mcep"],
        type=str, help="feature type")
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

    # read file list
    file_list = read_txt(args.feats)
    logging.info("number of utterances = %d" % len(file_list))

    # calculate statistics
    calc_stats(file_list, args)


if __name__ == "__main__":
    main()
