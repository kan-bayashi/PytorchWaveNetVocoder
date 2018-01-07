#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CALCULATE STATISTICS
"""
from __future__ import print_function

import argparse

import numpy as np
from sklearn.preprocessing import StandardScaler

from utils import read_hdf5, write_hdf5, read_txt


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--feats", default=None, required=True,
        help="name of the list of hdf5 files")
    parser.add_argument(
        "--stats", default=None, required=True,
        help="filename of hdf5 format")

    args = parser.parse_args()

    # read list and load all of features
    filenames = read_txt(args.feats)
    print("load feature vector file...", end="")
    feats = [read_hdf5(filename, "/feat_org") for filename in filenames]
    print("number of training utterances =", len(filenames))

    # calculate stats
    scaler = StandardScaler()
    [scaler.partial_fit(feat[:, 1:]) for feat in feats]

    # add uv term
    mean = np.zeros((feats[0].shape[1]))
    scale = np.ones((feats[0].shape[1]))
    mean[1:] = scaler.mean_
    scale[1:] = scaler.scale_

    # write to hdf5
    write_hdf5(args.stats, "/mean", mean)
    write_hdf5(args.stats, "/scale", scale)


if __name__ == "__main__":
    main()
