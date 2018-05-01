#!/bin/bash
############################################################
#        SCRIPT TO BUILD SI-CLOSE WAVENET VOCODER          #
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
# 3: apply noise shaping step
# 4: training step
# 5: decoding step
# 6: restore noise shaping step
# }}}
stage=0123456

#######################################
#          FEATURE SETTING            #
#######################################
# {{{
# shiftms: shift length in msec (default=5)
# fftl: fft length (default=1024)
# highpass_cutoff: highpass filter cutoff frequency (if 0, will not apply)
# mcep_dim: dimension of mel-cepstrum
# mcep_alpha: alpha value of mel-cepstrum
# mag: coefficient of noise shaping (default=0.5)
# n_jobs: number of parallel jobs
# }}}
feature_type=world
shiftms=5
fftl=1024
highpass_cutoff=70
fs=16000
mcep_dim=24
mcep_alpha=0.410
mag=0.5
n_jobs=10

#######################################
#          TRAINING SETTING           #
#######################################
# {{{
# n_gpus: number of gpus
# spks: target spekaers in arctic
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
spks=(bdl rms clb slt ksp jmk)
n_quantize=256
n_aux=28
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
use_noise_shaping=true
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
# }}}
outdir=
checkpoint=
config=
feats=
decode_batch_size=32

#######################################
#            OHTER SETTING            #
#######################################
ARCTIC_DB_ROOT=downloads
tag=

# parse options
. parse_options.sh

# set params
train=tr_"$(IFS=_; echo "${spks[*]}")"
eval=ev_"$(IFS=_; echo "${spks[*]}")"

# stop when error occured
set -e
# }}}


# STAGE 0 {{{
if echo ${stage} | grep -q 0 ;then
    echo "###########################################################"
    echo "#                 DATA PREPARATION STEP                   #"
    echo "###########################################################"
    if [ ! -e ${ARCTIC_DB_ROOT} ];then
        mkdir -p ${ARCTIC_DB_ROOT}
        cd ${ARCTIC_DB_ROOT}
        for id in bdl slt rms clb jmk ksp awb;do
            wget http://festvox.org/cmu_arctic/cmu_arctic/packed/cmu_us_${id}_arctic-0.95-release.tar.bz2
            tar xf cmu_us_${id}*.tar.bz2
        done
        rm ./*.tar.bz2
        cd ../
    fi
    [ ! -e "data/${train}" ] && mkdir -p "data/${train}"
    [ ! -e "data/${eval}" ] && mkdir -p "data/${eval}"
    [ -e "data/${train}/wav.scp" ] && rm "data/${train}/wav.scp"
    [ -e "data/${eval}/wav.scp" ] && rm "data/${eval}/wav.scp"
    for spk in "${spks[@]}";do
        find ${ARCTIC_DB_ROOT}/cmu_us_${spk}_arctic/wav -name "*.wav" \
            | sort | head -n 1028 >> "data/${train}/wav.scp"
        find ${ARCTIC_DB_ROOT}/cmu_us_${spk}_arctic/wav -name "*.wav" \
           | sort | tail -n 104 >> "data/${eval}/wav.scp"
    done
fi
# }}}


# STAGE 1 {{{
if echo ${stage} | grep -q 1 ;then
    echo "###########################################################"
    echo "#               FEATURE EXTRACTION STEP                   #"
    echo "###########################################################"
    nj=0
    for set in ${train} ${eval};do
        for spk in "${spks[@]}";do
            [ ! -e "exp/feature_extract/${set}" ] && mkdir -p "exp/feature_extract/${set}"
            # make scp of each speaker
            scp=exp/feature_extract/${set}/wav.${spk}.scp
            grep ${spk} "data/${set}/wav.scp" > "${scp}"

            # set f0 range
            minf0=$(awk '{print $1}' conf/${spk}.f0)
            maxf0=$(awk '{print $2}' conf/${spk}.f0)

            # feature extract
            ${train_cmd} --num-threads ${n_jobs} \
                "exp/feature_extract/feature_extract_${set}.${spk}.log" \
                feature_extract.py \
                    --waveforms "${scp}" \
                    --wavdir "wav/${set}/${spk}" \
                    --hdf5dir "hdf5/${set}/${spk}" \
                    --feature_type ${feature_type} \
                    --fs ${fs} \
                    --shiftms ${shiftms} \
                    --minf0 "${minf0}" \
                    --maxf0 "${maxf0}" \
                    --mcep_dim ${mcep_dim} \
                    --mcep_alpha ${mcep_alpha} \
                    --highpass_cutoff ${highpass_cutoff} \
                    --fftl ${fftl} \
                    --n_jobs ${n_jobs} &

            # update job counts
            nj=$(( nj + 1  ))
            if [ ! "${max_jobs}" -eq -1 ] && [ "${max_jobs}" -eq ${nj} ];then
                wait
                nj=0
            fi
        done
        wait

        # check the number of feature files
        n_wavs=$(wc -l "data/${set}/wav.scp")
        n_feats=$(find "hdf5/${set}" -name "*.h5" | wc -l)
        echo "${n_feats}/${n_wavs} files are successfully processed."

        # make scp files
        if [ ${highpass_cutoff} -eq 0 ];then
            cp "data/${set}/wav.scp" "data/${set}/wav_filtered.scp"
        else
            find "wav/${set}" -name "*.wav" | sort > "data/${set}/wav_filtered.scp"
        fi
        find "hdf5/${set}" -name "*.h5" | sort > "data/${set}/feats.scp"
    done
fi
# }}}


# STAGE 2 {{{
if echo ${stage} | grep -q 2 ;then
    echo "###########################################################"
    echo "#              CALCULATE STATISTICS STEP                  #"
    echo "###########################################################"
    ${train_cmd} "exp/calculate_statistics/calc_stats_${train}.log" \
        calc_stats.py \
            --feats "data/${train}/feats.scp" \
            --stats "data/${train}/stats.h5" \
            --feature_type ${feature_type}
    echo "statistics are successfully calculated."
fi
# }}}


# STAGE 3 {{{
if echo ${stage} | grep -q 3  && ${use_noise_shaping};then
    echo "###########################################################"
    echo "#                   NOISE SHAPING STEP                    #"
    echo "###########################################################"
    nj=0
    [ ! -e exp/noise_shaping ] && mkdir -p exp/noise_shaping
    for spk in "${spks[@]}";do
        # make scp of each speaker
        scp=exp/noise_shaping/wav_filtered.${spk}.scp
        grep "\/${spk}\/" "data/${train}/wav_filtered.scp" > ${scp}

        # apply noise shaping
        ${train_cmd} --num-threads ${n_jobs} \
            exp/noise_shaping/noise_shaping_apply.${spk}.log \
            noise_shaping.py \
                --waveforms ${scp} \
                --stats "data/${train}/stats.h5" \
                --writedir "wav_ns/${train}/${spk}" \
                --feature_type ${feature_type} \
                --fs ${fs} \
                --shiftms ${shiftms} \
                --fftl ${fftl} \
                --mcep_dim_start 2 \
                --mcep_dim_end $(( 2 + mcep_dim +1 )) \
                --mcep_alpha ${mcep_alpha} \
                --mag ${mag} \
                --inv true \
                --n_jobs ${n_jobs} &

        # update job counts
        nj=$(( nj + 1  ))
        if [ ! "${max_jobs}" -eq -1 ] && [ "${max_jobs}" -eq ${nj} ];then
            wait
            nj=0
        fi
    done
    wait

    # check the number of feature files
    n_wavs=$(wc -l "data/${train}/wav_filtered.scp")
    n_ns=$(find "wav_ns/${train}" -name "*.wav" | wc -l)
    echo "${n_ns}/${n_wavs} files are successfully processed."

    # make scp files
    find "wav_ns/${train}" -name "*.wav" | sort > "data/${train}/wav_ns.scp"
fi
# }}}


# STAGE 4 {{{
# set variables
if [ ! -n "${tag}" ];then
    expdir=exp/tr_arctic_16k_si_close_${feature_type}_nq${n_quantize}_na${n_aux}_nrc${n_resch}_nsc${n_skipch}_ks${kernel_size}_dp${dilation_depth}_dr${dilation_repeat}_lr${lr}_wd${weight_decay}_bl${batch_length}_bs${batch_size}
    if ${use_noise_shaping};then
        expdir=${expdir}_ns
    fi
    if ${use_upsampling};then
        expdir=${expdir}_up
    fi
else
    expdir=exp/tr_arctic_${tag}
fi
if echo ${stage} | grep -q 4 ;then
    echo "###########################################################"
    echo "#               WAVENET TRAINING STEP                     #"
    echo "###########################################################"
    if ${use_noise_shaping};then
        waveforms=data/${train}/wav_ns.scp
    else
        waveforms=data/${train}/wav_filtered.scp
    fi
    upsampling_factor=$(echo "${shiftms} * ${fs} / 1000" | bc)
    ${cuda_cmd} --gpu ${n_gpus} "${expdir}/log/train_${train}.log" \
        train.py \
            --n_gpus ${n_gpus} \
            --waveforms "${waveforms}" \
            --feats "data/${train}/feats.scp" \
            --stats "data/${train}/stats.h5" \
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


# STAGE 5 {{{
if echo ${stage} | grep -q 5 ;then
    echo "###########################################################"
    echo "#               WAVENET DECODING STEP                     #"
    echo "###########################################################"
    [ ! -n "${outdir}" ] && outdir=${expdir}/wav
    [ ! -n "${checkpoint}" ] && checkpoint=${expdir}/checkpoint-final.pkl
    [ ! -n "${config}" ] && config=${expdir}/model.conf
    [ ! -n "${feats}" ] && feats=data/${eval}/feats.scp
    [ ! -e exp/decoding ] && mkdir -p exp/decoding
    nj=0
    for spk in "${spks[@]}";do
        # make scp of each speaker
        scp=exp/decoding/feats.${spk}.scp
        grep "\/${spk}\/" "$feats" > ${scp}

        # decode
        ${cuda_cmd} --gpu ${n_gpus} "${outdir}/log/decode.${spk}.log" \
            decode.py \
                --n_gpus ${n_gpus} \
                --feats ${scp} \
                --stats "data/${train}/stats.h5" \
                --outdir "${outdir}/${spk}" \
                --checkpoint "${checkpoint}" \
                --config "${config}" \
                --fs ${fs} \
                --batch_size ${decode_batch_size} &

        # update job counts
        nj=$(( nj + 1  ))
        if [ ! "${max_jobs}" -eq -1 ] && [ "${max_jobs}" -eq ${nj} ];then
            wait
            nj=0
        fi
    done
    wait
fi
# }}}


# STAGE 6 {{{
if echo ${stage} | grep -q 6  && ${use_noise_shaping};then
    echo "###########################################################"
    echo "#             RESTORE NOISE SHAPING STEP                  #"
    echo "###########################################################"
    [ ! -n "${outdir}" ] && outdir=${expdir}/wav
    nj=0
    for spk in "${spks[@]}";do
        # make scp of each speaker
        scp=exp/noise_shaping/wav_generated.${spk}.scp
        find "${outdir}/${spk}" -name "*.wav" | grep "\/${spk}\/" | sort > ${scp}

        # restore noise shaping
        ${train_cmd} --num-threads ${n_jobs} \
            exp/noise_shaping/noise_shaping_restore.${spk}.log \
            noise_shaping.py \
                --waveforms ${scp} \
                --stats "data/${train}/stats.h5" \
                --writedir "${outdir}_restored/${spk}" \
                --feature_type ${feature_type} \
                --fs ${fs} \
                --shiftms ${shiftms} \
                --fftl ${fftl} \
                --mcep_dim_start 2 \
                --mcep_dim_end $(( 2 + mcep_dim +1 )) \
                --mcep_alpha ${mcep_alpha} \
                --mag ${mag} \
                --inv false \
                --n_jobs ${n_jobs} &

        # update job counts
        nj=$(( nj + 1  ))
        if [ ! "${max_jobs}" -eq -1 ] && [ "${max_jobs}" -eq ${nj} ];then
            wait
            nj=0
        fi
    done
    wait
fi
# }}}
