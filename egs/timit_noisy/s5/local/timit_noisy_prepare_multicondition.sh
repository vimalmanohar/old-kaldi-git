set -o pipefail 

dataid=train

. path.sh
. utils/parse_options.sh

if [ $# -ne 0 ]; then 
  echo "Incorrect number of options: $#; Expecting 0"
  exit 1
fi

mkdir -p data_noisy data_clean

utils/copy_data_dir.sh --spk-prefix clean- --utt-prefix clean- \
  data/${dataid}_clean data_noisy/${dataid}_clean || exit 1
utils/fix_data_dir.sh data_noisy/${dataid}_clean

utils/copy_data_dir.sh data_noisy/${dataid}_clean data_clean/${dataid}_clean || exit 1

# Make a map from the new utterance names to the old ones
paste data_noisy/${dataid}_clean/utt2spk data/${dataid}_clean/utt2spk \
  | perl -ane 'print "$F[0] $F[2]\n"' | sort -u -k1,1 \
  > data_noisy/${dataid}_clean/utt2uniq || exit 1

paste data_noisy/${dataid}_clean/utt2spk data/${dataid}_clean/utt2spk \
  | perl -ane 'print "$F[1] $F[3]\n"' | sort -u -k1,1 \
  > data_noisy/${dataid}_clean/spk2uniq || exit 1

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
      data/${dataid}_noisy_${noise_type}_snr_$snr \
      data_noisy/${dataid}_noisy_${noise_type}_snr_$snr || exit 1
    utils/fix_data_dir.sh data_noisy/${dataid}_noisy_${noise_type}_snr_$snr
    
    utils/copy_data_dir.sh --spk-prefix ${noise_type}-snr-$snr- \
      --utt-prefix ${noise_type}-snr-$snr- \
      data/${dataid}_clean \
      data_clean/${dataid}_clean_noisy_${noise_type}_snr_$snr || exit 1
    utils/fix_data_dir.sh data_clean/${dataid}_clean_noisy_${noise_type}_snr_$snr

    # Make a map from the new utterance names to the old ones
    paste data_noisy/${dataid}_noisy_${noise_type}_snr_$snr/utt2spk \
      data/${dataid}_noisy_${noise_type}_snr_$snr/utt2spk \
      | perl -ane 'print "$F[0] $F[2]\n"' | sort -u -k1,1 \
      > data_noisy/${dataid}_noisy_${noise_type}_snr_$snr/utt2uniq || exit 1
    
    paste data_noisy/${dataid}_noisy_${noise_type}_snr_$snr/utt2spk \
      data/${dataid}_noisy_${noise_type}_snr_$snr/utt2spk \
      | perl -ane 'print "$F[1] $F[3]\n"' | sort -u -k1,1 \
      > data_noisy/${dataid}_noisy_${noise_type}_snr_$snr/spk2uniq || exit 1

    noisy_dirs="$noisy_dirs data_noisy/${dataid}_noisy_${noise_type}_snr_$snr"
    clean_noisy_dirs="$clean_noisy_dirs data_clean/${dataid}_clean_noisy_${noise_type}_snr_$snr"
  done 4< conf/snr.list
done 3< conf/noisetypes.list

# Combine all the noise conditions together
utils/combine_data.sh --extra-files "utt2uniq spk2uniq" data_noisy/${dataid}_noisy$noisy_dirs || exit 1
utils/combine_data.sh --extra-files "utt2uniq spk2uniq" data_clean/${dataid}_clean_noisy$clean_noisy_dirs || exit 1
utils/fix_data_dir.sh data_noisy/${dataid}_noisy

# Combine the noisy data with the clean data
utils/combine_data.sh --extra-files "utt2uniq spk2uniq" \
  data_noisy/${dataid}_multi data_noisy/${dataid}_noisy data_noisy/${dataid}_clean || exit 1
utils/fix_data_dir.sh data_noisy/${dataid}_multi

utils/combine_data.sh --extra-files "utt2uniq spk2uniq" \
  data_clean/${dataid}_clean_multi data_clean/${dataid}_clean_noisy data_clean/${dataid}_clean || exit 1
utils/fix_data_dir.sh data_clean/${dataid}_clean_multi

