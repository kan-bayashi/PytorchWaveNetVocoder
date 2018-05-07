# Outline of recipes

This is the outline of recipes.

## Supported database

- [CMU Arctic database](http://www.festvox.org/cmu_arctic/): `egs/arctic`
- [LJ Speech database](https://keithito.com/LJ-Speech-Dataset/): `egs/ljspeech`
- [M-AILABS speech database](http://www.m-ailabs.bayern/en/the-mailabs-speech-dataset/): `egs/m-ailabs-speech`

## Type of recipe

`sd`: speaker-dependent model

- build speaker dependent model
- the speaker of training data is the same as that of evaluation data
- auxiliary features are based on World analysis
- noise shaping is applied

`si-open`: speaker-independent model in open condition

- build speaker independent model in spekaer-open condition
- the speakers of evaluation data does not include those of training data
- auxiliary features are based on World analysis
- noise shaping is applied

`si-close`: speaker-independent model in speaker-closed condition

- build speaker independent model in open condition
- the speakers of evaluation data includes those of training data
- auxiliary features are based on World analysis
- noise shaping is applied

`sd-melspc`: spekaer-denpendent model with mel-spectrogram

- build speaker dependent model with mel-spectrogram
- the speaker of training data is the same as that of evaluation data
- auxiliary features are mel-spectrogram
- noise shaping is **not** applied

## Flow of recipe

### Recipe with noise shaping (`sd/si-close/si-open`)

0. data preparation (`stage 0`)
1. auxiliary feature extraction (`stage 1`)
2. statistics calculation (`stage 2`)
3. noise shaping (`stage 3`)
4. WaveNet training (`stage 4`)
5. WaveNet decoding (`stage 5`)
6. restoring noise shaping (`stage 6`)

### Recipe without noise shaping (`sd-melspc`)

0. data preparation (`stage 0`)
1. auxiliary feature extraction (`stage 1`)
2. statistics calculation (`stage 2`)
4. WaveNet training (`stage 3`)
5. WaveNet decoding (`stage 4`)

## How-to-run

```bash
# change directory to one of the recipe
$ cd arctic/sd

# run the recipe
$ ./run.sh

# you can skip some stages (in this case only stage 4,5,6 will be conducted)
$ ./run.sh --stage 456

# you can also change hyperparameters via command line
$ ./run.sh --lr 1e-3 --batch_length 10000

# multi-gpu training / decoding are supported (batch size should be greater than #gpus)
$ ./run.sh --n_gpus 3 --batch_size 3
```

## Author

Tomoki Hayashi @ Nagoya University  
e-mail:hayashi.tomoki@g.sp.m.is.nagoya-u.ac.jp
