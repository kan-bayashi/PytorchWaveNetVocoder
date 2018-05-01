# PYTORCH-WAVENET-VOCODER

[![Build Status](https://travis-ci.org/kan-bayashi/PytorchWaveNetVocoder.svg?branch=master)](https://travis-ci.org/kan-bayashi/PytorchWaveNetVocoder)

This repository is the wavenet-vocoder implementation with pytorch.

![](https://github.com/kan-bayashi/WaveNetVocoderSamples/blob/master/figure/overview.bmp)

You can build above WaveNet vocoder using following datasets:
- [CMU Arctic database](http://www.festvox.org/cmu_arctic/): `egs/arctic`
- [LJ Speech database](https://keithito.com/LJ-Speech-Dataset/): `egs/ljspeech`
- [M-AILABS speech database](http://www.m-ailabs.bayern/en/the-mailabs-speech-dataset/): `egs/m-ailab-speech`

## News

### 2018/05/01: Major update

- Updated to be compatible with pytorch v0.4
- Updated to be able to use melspectrogram as auxiliary feature

Due to above update, some parts are changed (see below)

```
# -------------------- #
# feature path in hdf5 #
# -------------------- #
old -> new
/feat_org -> /world or /melspc
/feat_org -> no more saving extended featrue (it is replicated when loading)

# ----------------------- #
# statistics path in hdf5 #
# ----------------------- #
old -> new
/mean -> /world/mean or /melspc/mean
/scale -> /world/scale or /melspc/scale

# ----------------------- #
# new options in training #
# ----------------------- #
--feature_type: Auxiliary feature type (world or melspc)
--use_upsampling_layer: Flag to decide whether to use upsampling layer in WaveNet
--upsampling_factor: Changed to be alway needed because feature extension is performed in loading
```

Note that old model file `checkpoint-*.pkl` can be used, but it is necessary to modify `model.conf` file as follows.

```python
import torch
args = torch.load("old_model.conf")
args.use_upsampling_layer = True
args.feature_type = "world"
torch.save(args, "new_model.conf")
```

## Requirements
- python 3.6
- virtualenv
- cuda 8.0
- cndnn 6
- nccl 2.0+ (for the use of multi-gpus)

Recommend to use the GPU with 10GB> memory.

## Setup
```bash
$ git clone https://github.com/kan-bayashi/PytorchWaveNetVocoder.git
$ cd PytorchWaveNetVocoder/tools
$ make
```

## Run example
All examples are based on kaldi-style recipe.

```bash
# build SD model with world features
$ cd egs/arctic/sd
$ ./run.sh

# build SD model with mel-spectrogram
$ cd egs/arctic/sd-melspc
$ ./run.sh

# build SI-CLOSE model
$ cd egs/arctic/si-close
$ ./run.sh

# build SI-OPEN model
$ cd egs/arctic/si-open
$ ./run.sh

# Multi-GPU training and decoding
$ ./run.sh --n_gpus 3 --batch_size 3

# You can also change hyperparameters as follows
$ ./run.sh --n_gpus 3 --
```

If slurm is installed in your servers, you can run recipes with slurm.

```bash
$ cd egs/arctic/sd

# edit configuration
$ vim cmd.sh # please edit as follows
---
# for local
# export train_cmd="run.pl"
# export cuda_cmd="run.pl --gpu 1"

# for slurm (you can change configuration file "conf/slurm.conf")
export train_cmd="slurm.pl --config conf/slurm.conf"
export cuda_cmd="slurm.pl --gpu 1 --config conf/slurm.conf"
---

$ vim conf/slurm.conf # edit <your_partition_name>
---
command sbatch --export=PATH  --ntasks-per-node=1
option time=* --time $0
option mem=* --mem-per-cpu $0
option mem=0
option num_threads=* --cpus-per-task $0 --ntasks-per-node=1
option num_threads=1 --cpus-per-task 1  --ntasks-per-node=1
default gpu=0
option gpu=0 -p <your_partion_name>
option gpu=* -p <your_partion_name> --gres=gpu:$0 --time 10-00:00:00
---

# run the recipe
$ ./run.sh
```

Finally, you can get the generated wav files in `exp/train_*/wav_restored`.

## Use pre-trained model to decode your own data

To synthesize your own data, things what you need are as follows:

```
- checkpoint-final.pkl (model parameter file)
- model.conf (model configuration file)
- stats.h5 (feature statistics file)
- *.wav (your own wav file, should be 16000 Hz)
```

The procedure is as follows:

```bash
$ cd egs/arctic/si-close

# download pre-trained model which trained with 6 arctic speakers and world features
$ wget "https://www.dropbox.com/s/xt7qqmfgamwpqqg/si-close_lr1e-4_wd0_bs20k_ns_up.zip?dl=0" -O si-close_lr1e-4_wd0_bs20k_ns_up.zip

# unzip
$ unzip si-close_lr1e-4_wd0_bs20k_ns_up.zip

# make filelist of your own wav files
$ find <your_wav_dir> -name "*.wav" > wav.scp

# feature extraction
$ . ./path.sh
$ feature_extract.py \
    --waveforms wav.scp \
    --wavdir wav/test \
    --hdf5dir hdf5/test \
    --feature_type world \
    --fs 16000 \
    --shiftms 5 \
    --minf0 <set_appropriate_value> \
    --maxf0 <set_appropriate_value> \
    --mcep_dim 24 \
    --mcep_alpha 0.41 \
    --highpass_cutoff 70 \
    --fftl 1024 \
    --n_jobs 1

# make filelist of feature file
$ find hdf5/test -name "*.h5" > feats.scp

# decode with pre-trained model
$ decode.py \
    --feats feats.scp \
    --stats si-close_lr1e-4_wd0_bs20k_ns_up/stats.h5 \
    --outdir si-close_lr1e-4_wd0_bs20k_ns_up/wav \
    --checkpoint si-close_lr1e-4_wd0_bs20k_ns_up/checkpoint-final.pkl \
    --config si-close_lr1e-4_wd0_bs20k_ns_up/model.conf \
    --fs 16000 \
    --batch_size 32 \
    --n_gpus 1

# make filelist of generated wav file
$ find si-close_lr1e-4_wd0_bs20k_ns_up/wav -name "*.wav" > wav_generated.scp

# restore noise shaping
$ noise_shaping.py \
    --waveforms wav_generated.scp \
    --stats si-close_lr1e-4_wd0_bs20k_ns_up/stats.h5 \
    --writedir si-close_lr1e-4_wd0_bs20k_ns_up/wav_restored \
    --feature_type world \
    --fs 16000 \
    --shiftms 5 \
    --fftl 1024 \
    --mcep_dim_start 2 \
    --mcep_dim_end 27 \
    --mcep_alpha 0.41 \
    --mag 0.5 \
    --inv false \
    --n_jobs 1
```

Finally, you can hear the generated wav files in `si-close_lr1e-4_wd0_bs20k_ns_up/wav_restored`.

## Results

![](https://github.com/kan-bayashi/WaveNetVocoderSamples/blob/master/figure/mos.bmp)
Generated examples are available from [here](https://kan-bayashi.github.io/WaveNetVocoderSamples).

## References

Please cite the following articles.

```
@article{hayashi2018sp,
  title={複数話者WaveNetボコーダに関する調査}.
  author={林知樹 and 小林和弘 and 玉森聡 and 武田一哉 and 戸田智基},
  journal={電子情報通信学会技術研究報告},
  year={2018}
}
@inproceedings{hayashi2017multi,
  title={An Investigation of Multi-Speaker Training for WaveNet Vocoder},
  author={Hayashi, Tomoki and Tamamori, Akira and Kobayashi, Kazuhiro and Takeda, Kazuya and Toda, Tomoki},
  booktitle={Proc. ASRU 2017},
  year={2017}
}
@inproceedings{tamamori2017speaker,
  title={Speaker-dependent WaveNet vocoder},
  author={Tamamori, Akira and Hayashi, Tomoki and Kobayashi, Kazuhiro and Takeda, Kazuya and Toda, Tomoki},
  booktitle={Proceedings of Interspeech},
  pages={1118--1122},
  year={2017}
}
```

## Author
Tomoki Hayashi @ Nagoya University  
e-mail:hayashi.tomoki@g.sp.m.is.nagoya-u.ac.jp
