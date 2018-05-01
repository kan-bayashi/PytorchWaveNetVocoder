#!/bin/bash
############################################################
#           SCRIPT TO BUILD SD WAVENET VOCODER             #
############################################################

# Copyright 2017 Tomoki Hayashi (Nagoya University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh
. ./cmd.sh

# USER SETTINGS {{{
#######################################
#           STAGE SETTING             #
#######################################
# {{{
# 0: data preparation step
# 1: feature extraction step
# 2: statistics calculation step
# 3: training step
# 4: decoding step
# }}}
stage=01234

#######################################
#          FEATURE SETTING            #
#######################################
# {{{
# shiftms: shift length in msec (default=5)
# fftl: fft length (default=1024)
# highpass_cutoff: highpass filter cutoff frequency (if 0, will not apply)
# mspc_dim: dimension of mel-spectrogram
# n_jobs: number of parallel jobs
# }}}
feature_type=melspc
spk=elizabeth # judy (F) or mary (F) or elliot (M) or elizabeth (F)
shiftms=5
fftl=1024
highpass_cutoff=70
fs=16000
mspc_dim=80
n_jobs=10

#######################################
#          TRAINING SETTING           #
#######################################
# {{{
# n_gpus: number of gpus
# n_quantize: number of quantization
# n_aux: number of aux features
# n_resch: number of residual channels
# n_skipch: number of skip channels
# dilation_depth: dilation depth (e.g. if set 10, max dilation = 2^(10-1))
# dilation_repeat: number of dilation repeats
# kernel_size: kernel size of dilated convolution
# lr: learning rate
# weight_decay: weight decay coef
# iters: number of iterations
# batch_length: batch length
# batch_size: batch size
# checkpoints: save model per this number
# use_upsampling: true or false
# use_noise_shaping: true or false
# use_speaker_code: true or false
# resume: checkpoint to resume
# }}}
n_gpus=1
n_quantize=256
n_aux=80
n_resch=512
n_skipch=256
dilation_depth=10
dilation_repeat=3
kernel_size=2
lr=1e-4
weight_decay=0.0
iters=200000
batch_length=20000
batch_size=1
checkpoints=10000
use_upsampling=true
use_speaker_code=false
resume=

#######################################
#          DECODING SETTING           #
#######################################
# {{{
# outdir: directory to save decoded wav dir (if not set, will automatically set)
# checkpoint: full path of model to be used to decode (if not set, final model will be used)
# config: model configuration file (if not set, will automatically set)
# feats: list or directory of feature files
# n_gpus: number of gpus to decode
# }}}
outdir=
checkpoint=
config=
feats=
decode_batch_size=32

#######################################
#            OHTER SETTING            #
#######################################
DB_ROOT=downloads
tag=

# parse options
. parse_options.sh

# set params
train=tr_${spk}
eval=ev_${spk}

# stop when error occured
set -e
# }}}


# STAGE 0 {{{
if echo ${stage} | grep -q 0; then
    echo "###########################################################"
    echo "#                 DATA PREPARATION STEP                   #"
    echo "###########################################################"
    if [ ! -e ${DB_ROOT} ];then
        mkdir -p ${DB_ROOT}
        cd ${DB_ROOT}
        wget -O en_US.tgz http://www.m-ailabs.bayern/?ddownload=411
        wget -O en_UK.tgz http://www.m-ailabs.bayern/?ddownload=412
        tar xzvf ./*.tgz
        rm ./*.tgz
        cd ../
    fi
    [ ! -e data/${train} ] && mkdir -p data/${train}
    [ ! -e data/${eval} ] && mkdir -p data/${eval}
    if [ ${spk} = "elizabeth" ]; then
        find ${DB_ROOT}/en_UK/by_book/female/elizabeth_klett -name "*.wav" \
           | sort | grep -v "wives_and_daughters_60_" > data/${train}/wav.scp
        find ${DB_ROOT}/en_UK/by_book/female/elizabeth_klett -name "*.wav" \
           | sort | grep "wives_and_daughters_60_" > data/${eval}/wav.scp
    elif [ ${spk} = "judy" ]; then
        find ${DB_ROOT}/en_US/by_book/female/judy_bieber -name "*.wav" \
           | sort | grep -v "the_sea_faries_22_" > data/${train}/wav.scp
        find ${DB_ROOT}/en_US/by_book/female/judy_bieber -name "*.wav" \
           | sort | grep "the_sea_faries_22_" > data/${eval}/wav.scp
    elif [ ${spk} = "mary" ]; then
        find ${DB_ROOT}/en_US/by_book/female/mary_ann -name "*.wav" \
           | sort | grep -v "northandsouth_52_" > data/${train}/wav.scp
        find ${DB_ROOT}/en_US/by_book/female/mary_ann -name "*.wav" \
           | sort | grep "northandsouth_52_" > data/${eval}/wav.scp
    elif [ ${spk} = "elliot" ]; then
        find ${DB_ROOT}/en_US/by_book/male/elliot_miller -name "*.wav" \
           | sort | grep -v "silent_bullet_13_" > data/${train}/wav.scp
        find ${DB_ROOT}/en_US/by_book/male/elliot_miller -name "*.wav" \
           | sort | grep "silent_bullet_13_" > data/${eval}/wav.scp
    else
        echo "ERROR: spk should be selected from elizabeth, judy, mary, and elliot"
        exit 1
    fi
fi
# }}}


# STAGE 1 {{{
if echo ${stage} | grep -q 1; then
    echo "###########################################################"
    echo "#               FEATURE EXTRACTION STEP                   #"
    echo "###########################################################"
    for set in ${train} ${eval};do
        # training data feature extraction
        ${train_cmd} --num-threads ${n_jobs} exp/feature_extract/feature_extract_${set}.log \
            feature_extract.py \
                --waveforms data/${set}/wav.scp \
                --wavdir wav/${set} \
                --hdf5dir hdf5/${set} \
                --feature_type ${feature_type} \
                --fs ${fs} \
                --shiftms ${shiftms} \
                --mspc_dim ${mspc_dim} \
                --highpass_cutoff ${highpass_cutoff} \
                --fftl ${fftl} \
                --n_jobs ${n_jobs}

        # check the number of feature files
        n_wavs=$(wc -l data/${set}/wav.scp)
        n_feats=$(find hdf5/${set} -name "*.h5" | wc -l)
        echo "${n_feats}/${n_wavs} files are successfully processed."

        # make scp files
        if [ ${highpass_cutoff} -eq 0 ];then
            cp data/${set}/wav.scp data/${set}/wav_filtered.scp
        else
            find wav/${set} -name "*.wav" | sort > data/${set}/wav_filtered.scp
        fi
        find hdf5/${set} -name "*.h5" | sort > data/${set}/feats.scp
    done
fi
# }}}


# STAGE 2 {{{
if echo ${stage} | grep -q 2; then
    echo "###########################################################"
    echo "#              CALCULATE STATISTICS STEP                  #"
    echo "###########################################################"
    ${train_cmd} exp/calculate_statistics/calc_stats_${train}.log \
        calc_stats.py \
            --feats data/${train}/feats.scp \
            --stats data/${train}/stats.h5 \
            --feature_type ${feature_type}
    echo "statistics are successfully calculated."
fi
# }}}


# STAGE 3 {{{
# set variables
if [ ! -n "${tag}" ];then
    expdir=exp/tr_mai_16k_sd_${feature_type}_${spk}_nq${n_quantize}_na${n_aux}_nrc${n_resch}_nsc${n_skipch}_ks${kernel_size}_dp${dilation_depth}_dr${dilation_repeat}_lr${lr}_wd${weight_decay}_bl${batch_length}_bs${batch_size}
    if ${use_upsampling};then
        expdir=${expdir}_up
    fi
else
    expdir=exp/tr_mai_16k_${tag}
fi
if echo ${stage} | grep -q 3; then
    echo "###########################################################"
    echo "#               WAVENET TRAINING STEP                     #"
    echo "###########################################################"
    waveforms=data/${train}/wav_filtered.scp
    upsampling_factor=$(echo "${shiftms} * ${fs} / 1000" | bc)
    ${cuda_cmd} --gpu ${n_gpus} "${expdir}/log/${train}.log" \
        train.py \
            --n_gpus ${n_gpus} \
            --waveforms ${waveforms} \
            --feats data/${train}/feats.scp \
            --stats data/${train}/stats.h5 \
            --expdir "${expdir}" \
            --feature_type ${feature_type} \
            --n_quantize ${n_quantize} \
            --n_aux ${n_aux} \
            --n_resch ${n_resch} \
            --n_skipch ${n_skipch} \
            --dilation_depth ${dilation_depth} \
            --dilation_repeat ${dilation_repeat} \
            --kernel_size ${kernel_size} \
            --lr ${lr} \
            --weight_decay ${weight_decay} \
            --iters ${iters} \
            --batch_length ${batch_length} \
            --batch_size ${batch_size} \
            --checkpoints ${checkpoints} \
            --upsampling_factor "${upsampling_factor}" \
            --use_upsampling_layer ${use_upsampling} \
            --use_speaker_code ${use_speaker_code} \
            --resume "${resume}"
fi
# }}}


# STAGE 4 {{{
if echo ${stage} | grep -q 4; then
    echo "###########################################################"
    echo "#               WAVENET DECODING STEP                     #"
    echo "###########################################################"
    [ ! -n "${outdir}" ] && outdir=${expdir}/wav
    [ ! -n "${checkpoint}" ] && checkpoint=${expdir}/checkpoint-final.pkl
    [ ! -n "${config}" ] && config=${expdir}/model.conf
    [ ! -n "${feats}" ] && feats=data/${eval}/feats.scp
    ${cuda_cmd} --gpu ${n_gpus} "${outdir}"/log/decode.log \
        decode.py \
            --n_gpus ${n_gpus} \
            --feats ${feats} \
            --stats data/${train}/stats.h5 \
            --outdir "${outdir}" \
            --checkpoint "${checkpoint}" \
            --config "${config}" \
            --fs ${fs} \
            --batch_size ${decode_batch_size}
fi
# }}}
