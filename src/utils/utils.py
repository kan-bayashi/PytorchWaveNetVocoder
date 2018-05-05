# -*- coding: utf-8 -*-

# Copyright 2017 Tomoki Hayashi (Nagoya University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

from __future__ import division

import fnmatch
import logging
import os
import sys
import threading

import h5py
import numpy as np

from numpy.matlib import repmat


def check_hdf5(hdf5_name, hdf5_path):
    """FUNCTION TO CHECK HDF5 EXISTENCE

    Args:
        hdf5_name (str): filename of hdf5 file
        hdf5_path (str): dataset name in hdf5 file

    Return:
        (bool): dataset exists then return true
    """
    if not os.path.exists(hdf5_name):
        return False
    else:
        with h5py.File(hdf5_name, "r") as f:
            if hdf5_path in f:
                return True
            else:
                return False


def read_hdf5(hdf5_name, hdf5_path):
    """FUNCTION TO READ HDF5 DATASET

    Args:
        hdf5_name (str): filename of hdf5 file
        hdf5_path (str): dataset name in hdf5 file

    Return:
        dataset values
    """
    if not os.path.exists(hdf5_name):
        logging.error("there is no such a hdf5 file (%s)." % hdf5_name)
        sys.exit(1)

    hdf5_file = h5py.File(hdf5_name, "r")

    if hdf5_path not in hdf5_file:
        logging.error("there is no such a data in hdf5 file. (%s)" % hdf5_path)
        sys.exit(1)

    hdf5_data = hdf5_file[hdf5_path].value
    hdf5_file.close()

    return hdf5_data


def shape_hdf5(hdf5_name, hdf5_path):
    """FUNCTION TO GET HDF5 DATASET SHAPE

    Args:
        hdf5_name (str): filename of hdf5 file
        hdf5_path (str): dataset name in hdf5 file

    Return:
        (tuple): shape of dataset
    """
    if check_hdf5(hdf5_name, hdf5_path):
        with h5py.File(hdf5_name, "r") as f:
            hdf5_shape = f[hdf5_path].shape
        return hdf5_shape
    else:
        logging.error("there is no such a file or dataset")
        sys.exit(1)


def write_hdf5(hdf5_name, hdf5_path, write_data, is_overwrite=True):
    """FUNCTION TO WRITE DATASET TO HDF5

    Args :
        hdf5_name (str): hdf5 dataset filename
        hdf5_path (str): dataset path in hdf5
        write_data (ndarray): data to write
        is_overwrite (bool): flag to decide whether to overwrite dataset
    """
    # convert to numpy array
    write_data = np.array(write_data)

    # check folder existence
    folder_name, _ = os.path.split(hdf5_name)
    if not os.path.exists(folder_name) and len(folder_name) != 0:
        os.makedirs(folder_name)

    # check hdf5 existence
    if os.path.exists(hdf5_name):
        # if already exists, open with r+ mode
        hdf5_file = h5py.File(hdf5_name, "r+")
        # check dataset existence
        if hdf5_path in hdf5_file:
            if is_overwrite:
                logging.warn("dataset in hdf5 file already exists.")
                logging.warn("recreate dataset in hdf5.")
                hdf5_file.__delitem__(hdf5_path)
            else:
                logging.error("dataset in hdf5 file already exists.")
                logging.error("if you want to overwrite, please set is_overwrite = True.")
                hdf5_file.close()
                sys.exit(1)
    else:
        # if not exists, open with w mode
        hdf5_file = h5py.File(hdf5_name, "w")

    # write data to hdf5
    hdf5_file.create_dataset(hdf5_path, data=write_data)
    hdf5_file.flush()
    hdf5_file.close()


def find_files(directory, pattern="*.wav", use_dir_name=True):
    """FUNCTION TO FIND FILES RECURSIVELY

    Args:
        directory (str): root directory to find
        pattern (str): query to find
        use_dir_name (bool): if False, directory name is not included

    Return:
        (list): list of found filenames
    """
    files = []
    for root, dirnames, filenames in os.walk(directory, followlinks=True):
        for filename in fnmatch.filter(filenames, pattern):
            files.append(os.path.join(root, filename))
    if not use_dir_name:
        files = [file_.replace(directory + "/", "") for file_ in files]
    return files


def read_txt(file_list):
    """FUNCTION TO READ TXT FILE

    Arg:
        file_list (str): txt file filename

    Return:
        (list): list of read lines
    """
    with open(file_list, "r") as f:
        filenames = f.readlines()
    return [filename.replace("\n", "") for filename in filenames]


class BackgroundGenerator(threading.Thread):
    """BACKGROUND GENERATOR

    reference:
        https://stackoverflow.com/questions/7323664/python-generator-pre-fetch

    Args:
        generator (object): generator instance
        max_prefetch (int): max number of prefetch
    """

    def __init__(self, generator, max_prefetch=1):
        threading.Thread.__init__(self)
        if sys.version_info.major == 2:
            from Queue import Queue
        else:
            from queue import Queue
        self.queue = Queue(max_prefetch)
        self.generator = generator
        self.daemon = True
        self.start()

    def run(self):
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def next(self):
        next_item = self.queue.get()
        if next_item is None:
            raise StopIteration
        return next_item

    def __next__(self):
        return self.next()

    def __iter__(self):
        return self


class background(object):
    """BACKGROUND GENERATOR DECORATOR"""

    def __init__(self, max_prefetch=1):
        self.max_prefetch = max_prefetch

    def __call__(self, gen):
        def bg_generator(*args, **kwargs):
            return BackgroundGenerator(gen(*args, **kwargs))
        return bg_generator


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
