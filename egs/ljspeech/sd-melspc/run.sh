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
# 3: noise shaping step
# 4: training step
# 5: decoding step
# 6: restoring noise shaping step
# }}}
stage=0123456

#######################################
#          FEATURE SETTING            #
#######################################
# {{{
# shiftms: shift length in msec (default=5)
# fftl: fft length (default=1024)
# highpass_cutoff: highpass filter cutoff frequency (if 0, will not apply)
# mspc_dim: dimension of mel-spectrogram
# mcep_dim: dimension of mel-cepstrum (only used for noise shaping)
# mcep_alpha: all pass filter constant (only used for noise shaping)
# mag: coefficient of noise shaping (default=0.5)
# n_jobs: number of parallel jobs
# }}}
feature_type=melspc
shiftms=5
fftl=1024
highpass_cutoff=70
fs=22050
mspc_dim=80
mcep_dim=35
mcep_alpha=0.455
mag=0.5
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
# resume: checkpoint to resume
# }}}
n_gpus=1
n_quantize=256
n_aux=80
n_resch=512
n_skipch=256
dilation_depth=10
dilation_repeat=3
kernel_size=3
lr=1e-4
weight_decay=0.0
iters=200000
batch_length=15000
batch_size=1
checkpoints=10000
use_upsampling=true
use_noise_shaping=true
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
decode_batch_size=16

#######################################
#            OHTER SETTING            #
#######################################
LJSPEECH_DB_ROOT=downloads
tag=

# parse options
. parse_options.sh

# set params
train=tr
eval=ev

# stop when error occured
set -e
# }}}


# STAGE 0 {{{
if echo ${stage} | grep -q 0; then
    echo "###########################################################"
    echo "#                 DATA PREPARATION STEP                   #"
    echo "###########################################################"
    if [ ! -e ${LJSPEECH_DB_ROOT} ];then
        mkdir -p ${LJSPEECH_DB_ROOT}
        cd ${LJSPEECH_DB_ROOT}
        wget http://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
        tar -vxf ./*.tar.bz2
        rm ./*.tar.bz2
        cd ../
    fi
    [ ! -e data/${train} ] && mkdir -p data/${train}
    find ${LJSPEECH_DB_ROOT}/LJSpeech-1.1/wavs -name "*.wav" \
        | sort | grep -v LJ050 > data/${train}/wav.scp
    [ ! -e data/${eval} ] && mkdir -p data/${eval}
    find ${LJSPEECH_DB_ROOT}/LJSpeech-1.1/wavs -name "*.wav" \
       | sort | grep LJ050 > data/${eval}/wav.scp
fi
# }}}


# STAGE 1 {{{
if echo ${stage} | grep -q 1; then
    echo "###########################################################"
    echo "#               FEATURE EXTRACTION STEP                   #"
    echo "###########################################################"
    for set in ${train} ${eval};do
        # training data feature extraction
        ${train_cmd} --num-threads ${n_jobs} exp/feature_extract/feature_extract_${feature_type}_${set}.log \
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

        # extract stft-baed mel-cepstrum for noise shaping
        if [ ${set} = ${train} ] && ${use_noise_shaping};then
            ${train_cmd} --num-threads ${n_jobs} exp/feature_extract/feature_extract_mcep_${set}.log \
                feature_extract.py \
                    --waveforms data/${set}/wav.scp \
                    --wavdir wav/${set} \
                    --hdf5dir hdf5/${set} \
                    --feature_type mcep \
                    --fs ${fs} \
                    --shiftms ${shiftms} \
                    --mcep_dim ${mcep_dim} \
                    --mcep_alpha ${mcep_alpha} \
                    --highpass_cutoff ${highpass_cutoff} \
                    --save_wav false \
                    --fftl ${fftl} \
                    --n_jobs ${n_jobs}
        fi

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
    ${train_cmd} exp/calculate_statistics/calc_stats_${feature_type}_${train}.log \
        calc_stats.py \
            --feats data/${train}/feats.scp \
            --stats data/${train}/stats.h5 \
            --feature_type ${feature_type}
    if ${use_noise_shaping};then
        ${train_cmd} exp/calculate_statistics/calc_stats_mcep_${train}.log \
            calc_stats.py \
                --feats data/${train}/feats.scp \
                --stats data/${train}/stats.h5 \
                --feature_type mcep
    fi
    echo "statistics are successfully calculated."
fi
# }}}


# STAGE 3 {{{
if echo ${stage} | grep -q 3 && ${use_noise_shaping}; then
    echo "###########################################################"
    echo "#                   NOISE SHAPING STEP                    #"
    echo "###########################################################"
    ${train_cmd} --num-threads ${n_jobs} exp/noise_shaping/noise_shaping_apply_mcep_${train}.log \
        noise_shaping.py \
            --waveforms data/${train}/wav_filtered.scp \
            --stats data/${train}/stats.h5 \
            --writedir wav_ns/${train} \
            --feature_type mcep \
            --fs ${fs} \
            --shiftms ${shiftms} \
            --fftl ${fftl} \
            --mcep_alpha ${mcep_alpha} \
            --mag ${mag} \
            --inv true \
            --n_jobs ${n_jobs}

    # check the number of feature files
    n_wavs=$(wc -l data/${train}/wav_filtered.scp)
    n_ns=$(find wav_ns/${train} -name "*.wav" | wc -l)
    echo "${n_ns}/${n_wavs} files are successfully processed."

    # make scp files
    find wav_ns/${train} -name "*.wav" | sort > data/${train}/wav_ns.scp
fi
# }}}


# STAGE 4 {{{
# set variables
if [ ! -n "${tag}" ];then
    expdir=exp/tr_ljspeech_22k_sd_${feature_type}_nq${n_quantize}_na${n_aux}_nrc${n_resch}_nsc${n_skipch}_ks${kernel_size}_dp${dilation_depth}_dr${dilation_repeat}_lr${lr}_wd${weight_decay}_bl${batch_length}_bs${batch_size}
    if ${use_noise_shaping};then
        expdir=${expdir}_ns
    fi
    if ${use_upsampling};then
        expdir=${expdir}_up
    fi
else
    expdir=exp/tr_ljspeech_22k_${tag}
fi
if echo ${stage} | grep -q 4; then
    echo "###########################################################"
    echo "#               WAVENET TRAINING STEP                     #"
    echo "###########################################################"
    if ${use_noise_shaping};then
        waveforms=data/${train}/wav_ns.scp
    else
        waveforms=data/${train}/wav_filtered.scp
    fi
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
            --resume "${resume}"
fi
# }}}


# STAGE 5 {{{
if echo ${stage} | grep -q 5; then
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


# STAGE 6 {{{
if echo ${stage} | grep -q 6 && ${use_noise_shaping}; then
    echo "###########################################################"
    echo "#             RESTORE NOISE SHAPING STEP                  #"
    echo "###########################################################"
    [ ! -n "${outdir}" ] && outdir=${expdir}/wav
    find "${outdir}" -name "*.wav" | sort > data/${eval}/wav_generated.scp
    ${train_cmd} --num-threads ${n_jobs} exp/noise_shaping/noise_shaping_mcep_${eval}.log \
        noise_shaping.py \
            --waveforms data/${eval}/wav_generated.scp \
            --stats data/${train}/stats.h5 \
            --writedir "${outdir}_restored" \
            --feature_type mcep \
            --fs ${fs} \
            --shiftms ${shiftms} \
            --fftl ${fftl} \
            --mcep_alpha ${mcep_alpha} \
            --mag ${mag} \
            --n_jobs ${n_jobs} \
            --inv false
fi
# }}}
