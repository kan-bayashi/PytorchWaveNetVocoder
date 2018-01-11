# PYTORCH-WAVENET-VOCODER

This repository is the wavenet-vocoder implementation with pytorch.  

![](https://github.com/kan-bayashi/WaveNetVocoderSamples/blob/master/figure/overview.bmp)

## Requirements
- cuda 8.0
- python3.6
- virtualenv

Recommend to use the GPU with 10GB> memory.  

## Setup
```bash
cd tools
make -j
```

## Run example
All examples are based on kaldi-style recipe.  
If you are using Slurm, you can change cmd.sh in the recipe.  
```
# run SD model
cd egs/arctic/sd
./run.sh

# run SI-CLOSE model
cd egs/arctic/si-close
./run.sh 

# run SI-OPEN model
cd egs/arctic/si-open
./run.sh
```

## Results
![](https://github.com/kan-bayashi/WaveNetVocoderSamples/blob/master/figure/mos.bmp)
Generated examples are available from [here](https://kan-bayashi.github.io/WaveNetVocoderSamples)  

## References
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
