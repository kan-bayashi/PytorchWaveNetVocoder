# PYTORCH-WAVENET-VOCODER

This repository is the wavenet-vocoder implementation with pytorch.  

![](https://github.com/kan-bayashi/WaveNetVocoderSamples/blob/master/figure/overview.bmp)

## Requirements
- cuda 8.0
- python 3.6
- virtualenv

Recommend to use the GPU with 10GB> memory.  

## Setup
```bash
$ git clone https://github.com/kan-bayashi/PytorchWaveNetVocoder.git
$ cd PytorchWaveNetVocoder/tools
$ make -j
```

## Run example
All examples are based on kaldi-style recipe.  
```bash
# build SD model
$ cd egs/arctic/sd
$ ./run.sh 

# build SI-CLOSE model
$ cd egs/arctic/si-close
$ ./run.sh 

# build SI-OPEN model
$ cd egs/arctic/si-open
$ ./run.sh
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

## Use pre-trained model

Download pre-trained model from [here](https://www.dropbox.com/s/ifq9xw6gh1o3tzt/si-close_lr1e-4_wd0_bs20k_ns_up.zip?dl=0).

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
