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
- noise shaping with world mel-cepstrum is applied

`si-open`: speaker-independent model in open condition

- build speaker independent model in spekaer-open condition
- the speakers of evaluation data does not include those of training data
- auxiliary features are based on World analysis
- noise shaping with world mel-cepstrum is applied

`si-close`: speaker-independent model in speaker-closed condition

- build speaker independent model in open condition
- the speakers of evaluation data includes those of training data
- auxiliary features are based on World analysis
- noise shaping with world mel-cepstrum is applied

`*-melspc`: model with mel-spectrogram

- build the model with mel-spectrogram
- auxiliary features are mel-spectrogram
- noise shaping with stft mel-cepstrum is applied

## Flow of recipe

0. data preparation (`stage 0`)
1. auxiliary feature extraction (`stage 1`)
2. statistics calculation (`stage 2`)
3. noise shaping (`stage 3`)
4. WaveNet training (`stage 4`)
5. WaveNet decoding (`stage 5`)
6. restoring noise shaping (`stage 6`)

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

## Run recipe with slurm

If slurm is installed in your servers, you can run recipes with slurm.

```bash
$ cd egs/arctic/sd

# edit configuration
$ vim cmd.sh
# please edit as follows
-- cmd.sh --
# for local
# export train_cmd="run.pl"
# export cuda_cmd="run.pl --gpu 1"

# for slurm (you can change configuration file "conf/slurm.conf")
export train_cmd="slurm.pl --config conf/slurm.conf"
export cuda_cmd="slurm.pl --gpu 1 --config conf/slurm.conf"

$ vim conf/slurm.conf
# edit <your_partition_name>
-- slurm.conf --
command sbatch --export=PATH  --ntasks-per-node=1
option time=* --time $0
option mem=* --mem-per-cpu $0
option mem=0
option num_threads=* --cpus-per-task $0 --ntasks-per-node=1
option num_threads=1 --cpus-per-task 1  --ntasks-per-node=1
default gpu=0
option gpu=0 -p <your_partion_name>
option gpu=* -p <your_partion_name> --gres=gpu:$0 --time 10-00:00:00

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

## Author

Tomoki Hayashi @ Nagoya University  
e-mail:hayashi.tomoki@g.sp.m.is.nagoya-u.ac.jp
