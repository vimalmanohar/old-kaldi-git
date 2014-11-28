#!/bin/bash

# Copyright 2014  Vimal Manohar

. ./cmd.sh 
[ -f path.sh ] && . ./path.sh
set -e
set -u
set -o pipefail

echo $*

mkdir -p exp/make_irm_targets
mkdir -p irm_targets

noisy_dirs=
while read noise_type <&3; do
  while read snr <&4; do

    # Make IRM targets for each noise condition that would be used to
    # train the IRM predictor neural net
    local/make_irm_targets.sh data_fbank/train_noisy_${noise_type}_snr_$snr \
      data_fbank/train_clean exp/make_irm_targets irm_targets || exit 1
    
    # Before combining the data from all the noise conditions together
    # we need to add a prefix to each utterance and speaker that will 
    # distinguish the corresponding noise conditions
    # A prefix of the form babble-snr-10 etc. is added.
    utils/copy_data_dir.sh --spk-prefix ${noise_type}-snr-$snr- \
      --utt-prefix ${noise_type}-snr-$snr- \
      data_fbank/train_noisy_${noise_type}_snr_$snr \
      data_noisy_fbank/train_noisy_${noise_type}_snr_$snr

    # The prefix is added to irm_targets.scp as well
    cat data_fbank/train_noisy_${noise_type}_snr_$snr/irm_targets.scp | \
      awk '{print "'${noise_type}'-snr-'$snr'-"$1" "$2}' | sort -k1,1 \
      > data_noisy_fbank/train_noisy_${noise_type}_snr_$snr/irm_targets.scp
    
    # A list of noisy data directories that are to be combined
    noisy_dirs="${noisy_dirs} data_noisy_fbank/train_noisy_${noise_type}_snr_$snr"
  done 4< snr.list
done 3< noisetypes.list

# Combine all the noise conditions together
utils/combine_data.sh data_noisy_fbank/train_noisy$noisy_dirs || exit 1

for x in `echo $noisy_dirs | tr ' ' '\n'`; do 
  [ -d $x ] && cat $x/irm_targets.scp
done | sort -k1,1 > data_noisy_fbank/train_noisy/irm_targets.scp

# Combine the noisy data with the clean data
utils/combine_data.sh data_noisy_fbank/train_multi data_noisy_fbank/train_noisy data_fbank/train_clean || exit 1
