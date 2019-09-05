#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pip
import sys

from distutils.version import LooseVersion
from setuptools import find_packages
from setuptools import setup


if LooseVersion(sys.version) < LooseVersion("3.6"):
    raise RuntimeError(
        "Python>=3.6 is required, "
        "but your Python is {}".format(sys.version))
if LooseVersion(pip.__version__) < LooseVersion("19"):
    raise RuntimeError(
        "pip>=19.0.0 is required, but your pip is {}. "
        "Try again after \"pip install -U pip\"".format(pip.__version__))

requirements = {
    "install": [
        "h5py>=2.8.0",
        "scikit-learn>=0.20.2",
        "librosa>=0.6.2",
        "soundfile>=0.10.2",
        "torch>=1.0.1",
        "torchvision>=0.2.2",
        "sprocket-vc>=0.18.2",
        "matplotlib>=3.0.3",
    ],
    "setup": [
        "numpy",
        "pytest-runner"
    ],
    "test": [
        "pytest>=3.3.0",
        "hacking==1.1.0",
        "autopep8==1.2.4",
    ]}
install_requires = requirements["install"]
setup_requires = requirements["setup"]
tests_require = requirements["test"]
extras_require = {k: v for k, v in requirements.items()
                  if k not in ["install", "setup"]}

dirname = os.path.dirname(__file__)
setup(name="wavenet_vocoder",
      version="0.1.1",
      url="http://github.com/kan-bayashi/PytorchWaveNetVocoder",
      author="Tomoki Hayashi",
      author_email="hayashi.tomoki@g.sp.m.is.nagoya-u.ac.jp",
      description="Pytorch WaveNet Vocoder",
      long_description=open(os.path.join(dirname, "README.md"),
                            encoding="utf-8").read(),
      license="Apache Software License",
      packages=find_packages(include="wavenet_vocoder*"),
      install_requires=install_requires,
      setup_requires=setup_requires,
      tests_require=tests_require,
      extras_require=extras_require,
      classifiers=[
          "Programming Language :: Python",
          "Programming Language :: Python :: 3.6",
          "Intended Audience :: Science/Research",
          "Operating System :: POSIX :: Linux",
          "License :: OSI Approved :: Apache Software License",
          "Topic :: Software Development :: Libraries :: Python Modules"],
      )
