set -o pipefail 

mkdir -p data_noisy data_clean

utils/copy_data_dir.sh --spk-prefix clean- --utt-prefix clean- \
  data/train_clean data_noisy/train_clean || exit 1
utils/fix_data_dir.sh data_noisy/train_clean

utils/copy_data_dir.sh data_noisy/train_clean data_clean/train_clean || exit 1

# Make a map from the new utterance names to the old ones
paste data_noisy/train_clean/utt2spk data/train_clean/utt2spk \
  | perl -ane 'print "$F[0] $F[2]\n"' | sort -u -k1,1 \
  > data_noisy/train_clean/utt2uniq || exit 1

paste data_noisy/train_clean/utt2spk data/train_clean/utt2spk \
  | perl -ane 'print "$F[1] $F[3]\n"' | sort -u -k1,1 \
  > data_noisy/train_clean/spk2uniq || exit 1

noisy_dirs=
clean_noisy_dirs=
while read noise_type <&3; do
  while read snr <&4; do
    # Before combining the data from all the noise conditions together
    # we need to add a prefix to each utterance and speaker that will 
    # distinguish the corresponding noise conditions
    # A prefix of the form babble-snr-10 etc. is added.
    utils/copy_data_dir.sh --spk-prefix ${noise_type}-snr-$snr- \
      --utt-prefix ${noise_type}-snr-$snr- \
      data/train_noisy_${noise_type}_snr_$snr \
      data_noisy/train_noisy_${noise_type}_snr_$snr || exit 1
    utils/fix_data_dir.sh data_noisy/train_noisy_${noise_type}_snr_$snr
    
    utils/copy_data_dir.sh --spk-prefix ${noise_type}-snr-$snr- \
      --utt-prefix ${noise_type}-snr-$snr- \
      data/train_clean \
      data_clean/train_clean_noisy_${noise_type}_snr_$snr || exit 1
    utils/fix_data_dir.sh data_clean/train_clean_noisy_${noise_type}_snr_$snr

    # Make a map from the new utterance names to the old ones
    paste data_noisy/train_noisy_${noise_type}_snr_$snr/utt2spk \
      data/train_noisy_${noise_type}_snr_$snr/utt2spk \
      | perl -ane 'print "$F[0] $F[2]\n"' | sort -u -k1,1 \
      > data_noisy/train_noisy_${noise_type}_snr_$snr/utt2uniq || exit 1
    
    paste data_noisy/train_noisy_${noise_type}_snr_$snr/utt2spk \
      data/train_noisy_${noise_type}_snr_$snr/utt2spk \
      | perl -ane 'print "$F[1] $F[3]\n"' | sort -u -k1,1 \
      > data_noisy/train_noisy_${noise_type}_snr_$snr/spk2uniq || exit 1

    noisy_dirs="$noisy_dirs data_noisy/train_noisy_${noise_type}_snr_$snr"
    clean_noisy_dirs="$clean_noisy_dirs data_clean/train_clean_noisy_${noise_type}_snr_$snr"
  done 4< conf/snr.list
done 3< conf/noisetypes.list

# Combine all the noise conditions together
utils/combine_data.sh --extra-files "utt2uniq spk2uniq" data_noisy/train_noisy$noisy_dirs || exit 1
utils/combine_data.sh --extra-files "utt2uniq spk2uniq" data_clean/train_clean_noisy$clean_noisy_dirs || exit 1
utils/fix_data_dir.sh data_noisy/train_noisy

# Combine the noisy data with the clean data
utils/combine_data.sh --extra-files "utt2uniq spk2uniq" \
  data_noisy/train_multi data_noisy/train_noisy data_noisy/train_clean || exit 1
utils/fix_data_dir.sh data_noisy/train_multi

utils/combine_data.sh --extra-files "utt2uniq spk2uniq" \
  data_clean/train_clean_multi data_clean/train_clean_noisy data_clean/train_clean || exit 1
utils/fix_data_dir.sh data_clean/train_clean_multi

