#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2017 Tomoki Hayashi (Nagoya University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

from __future__ import print_function

import argparse

import numpy as np

from sklearn.preprocessing import StandardScaler

from utils import read_hdf5
from utils import read_txt
from utils import write_hdf5


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--feats", default=None, required=True,
        help="name of the list of hdf5 files")
    parser.add_argument(
        "--stats", default=None, required=True,
        help="filename of hdf5 format")

    args = parser.parse_args()

    # read list and define scaler
    filenames = read_txt(args.feats)
    scaler = StandardScaler()
    print("number of training utterances =", len(filenames))

    # process over all of data
    for filename in filenames:
        feat = read_hdf5(filename, "/feat_org")
        scaler.partial_fit(feat[:, 1:])

    # add uv term
    mean = np.zeros((feat.shape[1]))
    scale = np.ones((feat.shape[1]))
    mean[1:] = scaler.mean_
    scale[1:] = scaler.scale_

    # write to hdf5
    write_hdf5(args.stats, "/mean", mean)
    write_hdf5(args.stats, "/scale", scale)


if __name__ == "__main__":
    main()
