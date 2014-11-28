#!/bin/bash

# Copyright 2014  Vimal Manohar

. ./cmd.sh 
[ -f path.sh ] && . ./path.sh
set -e
set -u
set -o pipefail

echo $*

train_nj=48
decode_nj=24

. utils/parse_options.sh

# Prepare data to train IRM nnet
false && local/timit_prepare_irm_data.sh

# Train IRM predictor neural net. Run this outside this script.
false && {
local/run_irm_nnet.sh --irm-scp data_noisy_fbank/train_noisy/irm_targets.scp \
  --datadir data_noisy_fbank/train_noisy
}

mkdir -p irm 
mkdir -p masked_fbank

# Make masked fbank features by feed-forward propagating the noisy features
# through the IRM predictor neural net and applying the mask.
local/make_masked_fbank.sh --cmd "$train_cmd" --nj $train_nj \
  data_noisy_fbank/train_multi \
  data_noisy_fbank/train_multi_masked exp/irm_nnet \
  exp/make_masked_fbank/train_multi irm masked_fbank
steps/compute_cmvn_stats.sh --fake data_noisy_fbank/train_multi_masked \
  exp/make_masked_fbank/train_multi masked_fbank

# Make masked fbank features for each of the test and dev set noise conditions
##
while read noise_type <&3; do
  while read snr <&4; do
    for x in dev test; do
      x=${x}_noisy_${noise_type}_snr_$snr
      local/make_masked_fbank.sh --cmd "$train_cmd" --nj $decode_nj \
        data_fbank/${x} data_fbank/${x}_masked exp/irm_nnet exp/make_masked_fbank/$x irm masked_fbank
      steps/compute_cmvn_stats.sh --fake data_fbank/${x}_masked exp/make_masked_fbank/$x masked_fbank
    done
  done 4< snr.list
done 3< noisetypes.list

for x in dev test; do
  x=${x}_clean
  local/make_masked_fbank.sh --cmd "$train_cmd" --nj $decode_nj \
    data_fbank/${x} data_fbank/${x}_masked exp/irm_nnet exp/make_masked_fbank/$x irm masked_fbank
  steps/compute_cmvn_stats.sh --fake data_fbank/${x}_masked exp/make_masked_fbank/$x masked_fbank
done
##

# Compute MFCC features from the noisy and masked fbank features for training ASR
for y in data_noisy_fbank/train_multi_masked data_noisy_fbank/train_multi; do
  x=`basename $y`
  local/make_mfcc_from_fbank.sh --cmd "$train_cmd" --nj $train_nj \
    ${y} data/${x} exp/make_fbank_mfcc/$x fbank_mfcc
  steps/compute_cmvn_stats.sh data/${x} exp/make_fbank_mfcc/$x fbank_mfcc 
done

# Compute concatenated MFCC features from the noisy and masked fbank features for training ASR
local/make_concat_feats.sh --cmd "$train_cmd" --nj $train_nj \
  data/train_multi_concat data/train_multi \
  data/train_multi_masked exp/make_fbank_mfcc/train_multi_concat fbank_mfcc
steps/compute_cmvn_stats.sh data/train_multi_concat exp/make_fbank_mfcc/train_multi_concat fbank_mfcc

# Compute noisy, masked and concat MFCC features for each noise conditions for test and dev speech
##
while read noise_type <&3; do
  while read snr <&4; do
    for x in dev test; do
      x=${x}_noisy_${noise_type}_snr_$snr
      local/make_mfcc_from_fbank.sh --cmd "$train_cmd" --nj $decode_nj \
        data_fbank/${x} data/${x} \
        exp/make_fbank_mfcc/$x fbank_mfcc
      steps/compute_cmvn_stats.sh data/${x} exp/make_fbank_mfcc/${x} fbank_mfcc
      local/make_mfcc_from_fbank.sh --cmd "$train_cmd" --nj $decode_nj \
        data_fbank/${x}_masked data/${x}_masked \
        exp/make_fbank_mfcc/${x}_masked fbank_mfcc
      steps/compute_cmvn_stats.sh data/${x}_masked exp/make_fbank_mfcc/${x}_masked fbank_mfcc
      local/make_concat_feats.sh --cmd "$train_cmd" --nj $decode_nj \
        data/${x}_concat data/${x} data/${x}_masked exp/make_fbank_mfcc/${x}_concat fbank_mfcc
      steps/compute_cmvn_stats.sh data/${x}_concat exp/make_fbank_mfcc/${x}_concat fbank_mfcc
    done
  done 4< snr.list
done 3< noisetypes.list

for x in dev test; do
  x=${x}_clean
  local/make_mfcc_from_fbank.sh --cmd "$train_cmd" --nj $decode_nj \
    data_fbank/${x} data/${x} \
    exp/make_fbank_mfcc/$x fbank_mfcc
  steps/compute_cmvn_stats.sh data/${x} exp/make_fbank_mfcc/${x} fbank_mfcc
  local/make_mfcc_from_fbank.sh --cmd "$train_cmd" --nj $decode_nj \
    data_fbank/${x}_masked data/${x}_masked \
    exp/make_fbank_mfcc/${x}_masked fbank_mfcc
  steps/compute_cmvn_stats.sh data/${x}_masked exp/make_fbank_mfcc/${x}_masked fbank_mfcc
  local/make_concat_feats.sh --cmd "$train_cmd" --nj $decode_nj \
    data/${x}_concat data/${x} data/${x}_masked exp/make_fbank_mfcc/${x}_concat fbank_mfcc
  steps/compute_cmvn_stats.sh data/${x}_concat exp/make_fbank_mfcc/${x}_concat fbank_mfcc
done
##
