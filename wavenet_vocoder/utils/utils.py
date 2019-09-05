# -*- coding: utf-8 -*-

# Copyright 2017 Tomoki Hayashi (Nagoya University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import fnmatch
import logging
import os
import sys
import threading

import h5py
import numpy as np

from numpy.matlib import repmat


def check_hdf5(hdf5_name, hdf5_path):
    """CHECK HDF5 EXISTENCE.

    Args:
        hdf5_name (str): Filename of hdf5 file.
        hdf5_path (str): Dataset name in hdf5 file.

    Returns:
        bool: Dataset exists then return True.

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
    """READ HDF5 DATASET.

    Args:
        hdf5_name (str): Filename of hdf5 file.
        hdf5_path (str): Dataset name in hdf5 file.

    Return:
        any: Dataset values.

    """
    if not os.path.exists(hdf5_name):
        logging.error("there is no such a hdf5 file (%s)." % hdf5_name)
        sys.exit(1)

    hdf5_file = h5py.File(hdf5_name, "r")

    if hdf5_path not in hdf5_file:
        logging.error("there is no such a data in hdf5 file. (%s)" % hdf5_path)
        sys.exit(1)

    hdf5_data = hdf5_file[hdf5_path][()]
    hdf5_file.close()

    return hdf5_data


def shape_hdf5(hdf5_name, hdf5_path):
    """GET HDF5 DATASET SHAPE.

    Args:
        hdf5_name (str): Filename of hdf5 file.
        hdf5_path (str): Dataset name in hdf5 file.

    Returns:
        (tuple): Shape of dataset.

    """
    if check_hdf5(hdf5_name, hdf5_path):
        with h5py.File(hdf5_name, "r") as f:
            hdf5_shape = f[hdf5_path].shape
        return hdf5_shape
    else:
        logging.error("there is no such a file or dataset")
        sys.exit(1)


def write_hdf5(hdf5_name, hdf5_path, write_data, is_overwrite=True):
    """WRITE DATASET TO HDF5.

    Args:
        hdf5_name (str): Hdf5 dataset filename.
        hdf5_path (str): Dataset path in hdf5.
        write_data (ndarray): Data to write.
        is_overwrite (bool): Whether to overwrite dataset.

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
                logging.warning("dataset in hdf5 file already exists.")
                logging.warning("recreate dataset in hdf5.")
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
    """FIND FILES RECURSIVELY.

    Args:
        directory (str): Root directory to find.
        pattern (str): Query to find.
        use_dir_name (bool): If False, directory name is not included.

    Returns:
        list: List of found filenames.

    """
    files = []
    for root, dirnames, filenames in os.walk(directory, followlinks=True):
        for filename in fnmatch.filter(filenames, pattern):
            files.append(os.path.join(root, filename))
    if not use_dir_name:
        files = [file_.replace(directory + "/", "") for file_ in files]
    return files


def read_txt(file_list):
    """READ TXT FILE.

    Args:
        file_list (str): TXT file filename.

    Returns:
        list: List of read lines.

    """
    with open(file_list, "r") as f:
        filenames = f.readlines()
    return [filename.replace("\n", "") for filename in filenames]


class BackgroundGenerator(threading.Thread):
    """BACKGROUND GENERATOR.

    Args:
        generator (object): Generator instance.
        max_prefetch (int): Max number of prefetch.

    References:
        https://stackoverflow.com/questions/7323664/python-generator-pre-fetch

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
        """STORE ITEMS IN QUEUE."""
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def next(self):
        """GET ITEM IN THE QUEUE."""
        next_item = self.queue.get()
        if next_item is None:
            raise StopIteration
        return next_item

    def __next__(self):
        return self.next()

    def __iter__(self):
        return self


class background(object):
    """BACKGROUND GENERATOR DECORATOR."""

    def __init__(self, max_prefetch=1):
        self.max_prefetch = max_prefetch

    def __call__(self, gen):
        def bg_generator(*args, **kwargs):
            return BackgroundGenerator(gen(*args, **kwargs))
        return bg_generator


def extend_time(feats, upsampling_factor):
    """EXTEND TIME RESOLUTION.

    Args:
        feats (ndarray): Feature vector with the shape (T, D).
        upsampling_factor (int): Upsampling_factor.

    Returns:
        (ndarray): Extended feats with the shape (upsampling_factor * T, D).

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
