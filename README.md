# PYTORCH-WAVENET-VOCODER

[![Build Status](https://travis-ci.org/kan-bayashi/PytorchWaveNetVocoder.svg?branch=master)](https://travis-ci.org/kan-bayashi/PytorchWaveNetVocoder)

This repository is the wavenet-vocoder implementation with pytorch.

![](https://kan-bayashi.github.io/WaveNetVocoderSamples/images/overview.bmp)

You can try the demo recipe in Google colab from now!

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kan-bayashi/INTERSPEECH19_TUTORIAL/blob/master/notebooks/wavenet_vocoder/wavenet_vocoder.ipynb)

## Key features

- Support kaldi-like recipe, easy to reproduce the results
- Support multi-gpu training / decoding
- Support world features / mel-spectrogram as auxiliary features
- Support recipes of three public databases

    - [CMU Arctic database](http://www.festvox.org/cmu_arctic/): `egs/arctic`
    - [LJ Speech database](https://keithito.com/LJ-Speech-Dataset/): `egs/ljspeech`
    - [M-AILABS speech database](http://www.m-ailabs.bayern/en/the-mailabs-speech-dataset/): `egs/m-ailabs-speech`

## Requirements

- python 3.6+
- virtualenv
- cuda 9.0+
- cndnn 7.1+
- nccl 2.0+ (for the use of multi-gpus)

Recommend to use the GPU with 10GB> memory.

## Setup

### A. Make virtualenv

```bash
$ git clone https://github.com/kan-bayashi/PytorchWaveNetVocoder.git
$ cd PytorchWaveNetVocoder/tools
$ make
```

### B. Install with pip

```
$ git clone https://github.com/kan-bayashi/PytorchWaveNetVocoder.git
$ cd PytorchWaveNetVocoder

# recommend to use with pytorch 1.0.1 because only tested on 1.0.1
$ pip install torch==1.0.1 torchvision==0.2.2
$ pip install -e .

# please make dummy activate file to suppress warning in the recipe
$ mkdir -p tools/venv/bin && touch tools/venv/bin/activate
```

## How-to-run

```bash
$ cd egs/arctic/sd
$ ./run.sh
```

See more detail of the recipes in [egs/README.md](egs/README.md).

## Results

You can listen to samples from [kan-bayashi/WaveNetVocoderSamples](https://kan-bayashi.github.io/WaveNetVocoderSamples/).

This is the subjective evaluation results using `arctic` recipe.

**Comparison between model type**
![](https://kan-bayashi.github.io/WaveNetVocoderSamples/images/mos.bmp)

**Effect of the amount of training data**
![](https://kan-bayashi.github.io/WaveNetVocoderSamples/images/mos_num_train.bmp)

If you want to listen more samples, please access our google drive from [here](https://drive.google.com/drive/folders/1zC1WDiMu4SOdc7UeOayoEe_79PdnPBu6?usp=sharing).

Here is the list of samples:
- `arctic_raw_16k`: original in arctic database
- `arctic_sd_16k_world`: sd model with world aux feats + noise shaping with world mcep
- `arctic_si-open_16k_world`: si-open model with world aux feats + noise shaping with world mcep
- `arctic_si-close_16k_world`: si-close model with world aux feats + noise shaping with world mcep
- `arctic_si-close_16k_melspc`: si-close model with mel-spectrogram aux feats
- `arctic_si-close_16k_melspc_ns`: si-close model with mel-spectrogram aux feats + noise shaping with stft mcep
- `ljspeech_raw_22.05k`: original in ljspeech database
- `ljspeech_sd_22.05k_world`: sd model with world aux feats + noise shaping with world mcep
- `ljspeech_sd_22.05k_melspc`: sd model with mel-spectrogram aux feats
- `ljspeech_sd_22.05k_melspc_ns`: sd model with mel-spectrogram aux feats + noise shaping with stft mcep
- `m-ailabs_raw_16k`: original in m-ailabs speech database
- `m-ailabs_sd_16k_melspc`: sd model with mel-spectrogram aux feats

## References

Please cite the following articles.

```
@inproceedings{tamamori2017speaker,
  title={Speaker-dependent WaveNet vocoder},
  author={Tamamori, Akira and Hayashi, Tomoki and Kobayashi, Kazuhiro and Takeda, Kazuya and Toda, Tomoki},
  booktitle={Proceedings of Interspeech},
  pages={1118--1122},
  year={2017}
}
@inproceedings{hayashi2017multi,
  title={An Investigation of Multi-Speaker Training for WaveNet Vocoder},
  author={Hayashi, Tomoki and Tamamori, Akira and Kobayashi, Kazuhiro and Takeda, Kazuya and Toda, Tomoki},
  booktitle={Proc. ASRU 2017},
  year={2017}
}
@article{hayashi2018sp,
  title={複数話者WaveNetボコーダに関する調査}.
  author={林知樹 and 小林和弘 and 玉森聡 and 武田一哉 and 戸田智基},
  journal={電子情報通信学会技術研究報告},
  year={2018}
}
```

## Author

Tomoki Hayashi @ Nagoya University  
e-mail:hayashi.tomoki@g.sp.m.is.nagoya-u.ac.jp
