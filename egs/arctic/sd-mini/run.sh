#!/bin/bash
############################################################
#         DEMO SCRIPT TO BUILD SD WAVENET VOCODER          #
############################################################

# Copyright 2017 Tomoki Hayashi (Nagoya University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;

# USER SETTINGS {{{
#######################################
#           STAGE SETTING             #
#######################################
stage=0123456
# 0: data preparation step
# 1: feature extraction step
# 2: statistics calculation step
# 3: apply noise shaping step
# 4: training step
# 5: decoding step
# 6: restore noise shaping step

#######################################
#          FEATURE SETTING            #
#######################################
feature_type=world     # world or melspc (in this recipe fixed to "world")
spk=slt                # target spekaer in arctic
minf0=""               # minimum f0 (if not set, conf/*.f0 will be used)
maxf0=""               # maximum f0 (if not set, conf/*.f0 will be used)
shiftms=5              # shift length in msec
fftl=1024              # fft length
highpass_cutoff=70     # highpass filter cutoff frequency (if 0, will not apply)
fs=16000               # sampling rate
mcep_dim=24            # dimension of mel-cepstrum
mcep_alpha=0.410       # alpha value of mel-cepstrum
use_noise_shaping=true # whether to use noise shaping
mag=0.5                # strength of noise shaping (0.0 < mag <= 1.0)
n_jobs=10              # number of parallel jobs

#######################################
#          TRAINING SETTING           #
#######################################
n_gpus=1                # number of gpus (default=1)
n_quantize=256          # number of quantization of waveform
n_aux=28                # number of auxliary features
n_resch=32              # number of residual channels
n_skipch=16             # number of skip channels
dilation_depth=5        # dilation depth (e.g. if set 10, max dilation = 2^(10-1))
dilation_repeat=1       # number of dilation repeats
kernel_size=2           # kernel size of dilated convolution
lr=1e-4                 # learning rate
weight_decay=0.0        # weight decay coef
iters=1000              # number of iterations
batch_length=10000      # batch length
batch_size=1            # batch size
checkpoint_interval=100 # save model per this number
use_upsampling=true     # whether to use upsampling layer
resume=""               # checkpoint paht to resume (Optional)

#######################################
#          DECODING SETTING           #
#######################################
outdir=""           # directory to save decoded wav dir (Optional)
checkpoint=""       # checkpoint path to be used for decoding (Optional)
config=""           # model configuration path (Optional)
stats=""            # statistics path (Optional
feats=""            # list or directory of feature files
decode_batch_size=4 # batch size in decoding

#######################################
#            OHTER SETTING            #
#######################################
download_dir=downloads # download directory to save corpus
download_url="https://drive.google.com/open?id=1NIia89CL2qqqDzNNc718wycRmI_jkLxR" # download URL of gooogle drive
tag="" # tag for network directory naming (Optional)

# This enable argparse-like parsing of the above variables e.g. ./run.sh --stage 0
. parse_options.sh || exit 1;

# check feature type
if [ ${feature_type} != "world" ]; then
    echo "This recipe does not support feature_type=\"melspc\"." 2>&1
    echo "Please try the egs/*/*-melspc." 2>&1
    exit 1;
fi

# set directory names
train=tr_${spk}
eval=ev_${spk}

# stop when error occured
set -euo pipefail
# }}}


# STAGE 0 {{{
if echo ${stage} | grep -q 0; then
    echo "###########################################################"
    echo "#                 DATA PREPARATION STEP                   #"
    echo "###########################################################"
    # download dataset
    if [ ! -e ${download_dir}/.done ];then
        download_from_google_drive.sh "${download_url}" ${download_dir} tar.gz
        touch ${download_dir}/.done
    fi
    # use first 32 utterances as training data
    [ ! -e data/${train} ] && mkdir -p data/${train}
    find ${download_dir}/cmu_us_${spk}_arctic_mini/wav -name "*.wav" \
        | sort | head -n 32 > data/${train}/wav.scp
    echo "making wav list for training is successfully done. (#training = $(wc -l < data/${train}/wav.scp))"

    # use next 4 utterances as evaluation data
    [ ! -e data/${eval} ] && mkdir -p data/${eval}
    find ${download_dir}/cmu_us_${spk}_arctic_mini/wav -name "*.wav" \
       | sort | tail -n 4 > data/${eval}/wav.scp
    echo "making wav list for evaluation is successfully done. (#evaluation = $(wc -l < data/${eval}/wav.scp))"
fi
# }}}


# STAGE 1 {{{
if echo ${stage} | grep -q 1; then
    echo "###########################################################"
    echo "#               FEATURE EXTRACTION STEP                   #"
    echo "###########################################################"
    [ ! -n "${minf0}" ] && minf0=$(awk '{print $1}' conf/${spk}.f0)
    [ ! -n "${maxf0}" ] && maxf0=$(awk '{print $2}' conf/${spk}.f0)
    [ ! -e exp/feature_extract ] && mkdir -p exp/feature_extract
    for set in ${train} ${eval};do
        [ "${set}" = "${train}" ] && save_wav=true || save_wav=false
        feature_extract.py \
            --waveforms data/${set}/wav.scp \
            --wavdir wav/${set} \
            --hdf5dir hdf5/${set} \
            --feature_type ${feature_type} \
            --fs ${fs} \
            --shiftms ${shiftms} \
            --minf0 "${minf0}" \
            --maxf0 "${maxf0}" \
            --mcep_dim ${mcep_dim} \
            --mcep_alpha ${mcep_alpha} \
            --highpass_cutoff ${highpass_cutoff} \
            --fftl ${fftl} \
            --save_wav ${save_wav} \
            --n_jobs ${n_jobs} 2>&1 | tee exp/feature_extract/feature_extract_${set}.log

        # check the number of feature files
        n_wavs=$(wc -l data/${set}/wav.scp)
        n_feats=$(find hdf5/${set} -name "*.h5" | wc -l)
        echo "${n_feats}/${n_wavs} files are successfully processed."

        # make scp files
        if [ ${highpass_cutoff} -eq 0 ];then
            cp data/${set}/wav.scp data/${set}/wav_filtered.scp
        elif ${save_wav}; then
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
    [ ! -e exp/calculate_statistics ] && mkdir -p exp/calculate_statistics
    calc_stats.py \
        --feats data/${train}/feats.scp \
        --stats data/${train}/stats.h5 \
        --feature_type ${feature_type} | tee exp/calculate_statistics/calc_stats_${train}.log
    echo "statistics are successfully calculated."
fi
# }}}


# STAGE 3 {{{
if echo ${stage} | grep -q 3 && ${use_noise_shaping}; then
    echo "###########################################################"
    echo "#                   NOISE SHAPING STEP                    #"
    echo "###########################################################"
    [ ! -e exp/noise_shaping ] && mkdir -p exp/noise_shaping
    noise_shaping.py \
        --waveforms data/${train}/wav_filtered.scp \
        --stats data/${train}/stats.h5 \
        --outdir wav_ns/${train} \
        --feature_type ${feature_type} \
        --fs ${fs} \
        --shiftms ${shiftms} \
        --mcep_dim_start 2 \
        --mcep_dim_end $(( 2 + mcep_dim +1 )) \
        --mcep_alpha ${mcep_alpha} \
        --mag ${mag} \
        --inv true \
        --n_jobs ${n_jobs} 2>&1 | tee exp/noise_shaping/noise_shaping_apply_${train}.log

    # check the number of feature files
    n_wavs=$(wc -l data/${train}/wav_filtered.scp)
    n_ns=$(find wav_ns/${train} -name "*.wav" | wc -l)
    echo "${n_ns}/${n_wavs} files are successfully processed."

    # make scp files
    find wav_ns/${train} -name "*.wav" | sort > data/${train}/wav_ns.scp
fi # }}}


# STAGE 4 {{{
# set variables
if [ ! -n "${tag}" ];then
    expdir=exp/tr_arctic_16k_sd_${feature_type}_${spk}_nq${n_quantize}_na${n_aux}_nrc${n_resch}_nsc${n_skipch}_ks${kernel_size}_dp${dilation_depth}_dr${dilation_repeat}_lr${lr}_wd${weight_decay}_bl${batch_length}_bs${batch_size}
    if ${use_noise_shaping};then
        expdir=${expdir}_ns
    fi
    if ${use_upsampling};then
        expdir=${expdir}_up
    fi
else
    expdir=exp/tr_arctic_${tag}
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
    [ ! -e ${expdir}/log ] && mkdir -p ${expdir}/log
    [ ! -e ${expdir}/stats.h5 ] && cp -v data/${train}/stats.h5 ${expdir}
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
        --checkpoint_interval ${checkpoint_interval} \
        --upsampling_factor "${upsampling_factor}" \
        --use_upsampling_layer ${use_upsampling} \
        --resume "${resume}" 2>&1 | tee -a ${expdir}/log/${train}.log
fi
# }}}


# STAGE 5 {{{
[ ! -n "${outdir}" ] && outdir=${expdir}/wav
[ ! -n "${checkpoint}" ] && checkpoint=${expdir}/checkpoint-final.pkl
[ ! -n "${config}" ] && config=$(dirname ${checkpoint})/model.conf
[ ! -n "${stats}" ] && stats=$(dirname ${checkpoint})/stats.h5
[ ! -n "${feats}" ] && feats=data/${eval}/feats.scp
if echo ${stage} | grep -q 5; then
    echo "###########################################################"
    echo "#               WAVENET DECODING STEP                     #"
    echo "###########################################################"
    [ ! -e ${outdir}/log ] && mkdir -p ${outdir}/log
    decode.py \
        --n_gpus ${n_gpus} \
        --feats ${feats} \
        --stats "${stats}" \
        --outdir "${outdir}" \
        --checkpoint "${checkpoint}" \
        --config "${config}" \
        --fs ${fs} \
        --batch_size ${decode_batch_size} 2>&1 | tee ${outdir}/log/decode.log
fi
# }}}


# STAGE 6 {{{
if echo ${stage} | grep -q 6 && ${use_noise_shaping}; then
    echo "###########################################################"
    echo "#             RESTORE NOISE SHAPING STEP                  #"
    echo "###########################################################"
    find "${outdir}" -name "*.wav" | sort > ${outdir}/wav.scp
    [ ! -e exp/noise_shaping ] && mkdir -p exp/noise_shaping
    noise_shaping.py \
        --waveforms ${outdir}/wav.scp \
        --stats "${stats}" \
        --outdir ${outdir}_restored \
        --fs ${fs} \
        --shiftms ${shiftms} \
        --n_jobs ${n_jobs} \
        --inv false 2>&1 | tee exp/noise_shaping/noise_shaping_restore_${eval}.log
fi
# }}}
