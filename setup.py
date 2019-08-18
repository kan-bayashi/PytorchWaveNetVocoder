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
        "ESPnet requires Python>=3.6, "
        "but your Python is {}".format(sys.version))
if LooseVersion(pip.__version__) < LooseVersion("19"):
    raise RuntimeError(
        "pip>=19.0.0 is required, but your pip is {}. "
        "Try again after \"pip install -U pip\"".format(pip.__version__))

requirements = {
    "install": [
        "h5py>=2.9.0",
        "scikit-learn>=0.20.2",
        "librosa>=0.6.2",
        "soundfile>=0.10.2",
        "torch==1.0.1",
        "torchvision>=0.2.1",
        "sprocket-vc>=0.18.2",
        "matplotlib>=3.1.0",
    ],
    "setup": [
        "numpy",
        "pytest-runner"
    ],
    "test": [
        "pytest>=3.3.0",
        "pytest-pythonpath>=0.7.1",
        "hacking>=1.0.0",
        "mock>=2.0.0",
        "autopep8>=1.3.3",
    ]}
install_requires = requirements["install"]
setup_requires = requirements["setup"]
tests_require = requirements["test"]
extras_require = {k: v for k, v in requirements.items()
                  if k not in ["install", "setup"]}

dirname = os.path.dirname(__file__)
setup(name="wavenet_vocoder",
      version="0.1.0",
      url="http://github.com/kan-bayashi/PytorchWaveNetVocoder",
      author="Tomoki Hayashi",
      author_email="hayashi.tomoki@g.sp.m.is.nagoya-u.ac.jp",
      description="Pytorch WaveNet Vocoder",
      long_description=open(os.path.join(dirname, "README.md"),
                            encoding="utf-8").read(),
      license="Apache Software License",
      packages=find_packages(include=["wavenet_vocoder*"], exclude=["*.pl", "*.sh"]),
      # #448: "scripts" is inconvenient for developping because they are copied
      # scripts=get_all_scripts("espnet/bin"),
      install_requires=install_requires,
      setup_requires=setup_requires,
      tests_require=tests_require,
      extras_require=extras_require,
      classifiers=[
          "Programming Language :: Python",
          "Programming Language :: Python :: 2.7",
          "Programming Language :: Python :: 3.6",
          "Intended Audience :: Science/Research",
          "Operating System :: POSIX :: Linux",
          "License :: OSI Approved :: Apache Software License",
          "Topic :: Software Development :: Libraries :: Python Modules"],
      )
