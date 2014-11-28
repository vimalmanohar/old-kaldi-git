. path.sh
. cmd.sh

set -e
set -o pipefail

for trainid in clean random; do

  if [ ! -f $exp/tri4_nnet_${trainid}/final.mdl ]; then
    local/run_nnet.sh data/train_${trainid}
  else

    for t in data/dev_clean data/test_clean data/dev_clean_masked data/test_clean_masked; do
      if [ ! -f $exp/tri4_nnet_${trainid}/decode_${t##*/}/.done ]; then
        $train_cmd local/run_nnet.decode.sh $t $exp/tri3/graph $exp/tri4_nnet_${trainid} &
      fi
    done

    while read noise_type <&3; do
      while read snr <&4; do
        for t in data/dev_noisy_${noise_type}_snr_${snr} data/test_noisy_${noise_type}_snr_${snr} data/dev_noisy_${noise_type}_snr_${snr}_masked data/test_noisy_${noise_type}_snr_${snr}_masked; do 
          #steps/compute_cmvn_stats.sh $t $exp/make_fbank_mfcc/`basename $t` fbank_mfcc
          #utils/fix_data_dir.sh $t
          if [ ! -f $exp/tri4_nnet_${trainid}/decode_${t##*/}/.done ]; then
            $train_cmd local/run_nnet.decode.sh $t $exp/tri3/graph $exp/tri4_nnet_${trainid} &
          fi
        done
      done 4< snr.list
    done 3< noisetypes.list
  fi
done

for trainid in train_random_masked; do

  if [ ! -f $exp/tri4_nnet_${trainid}/final.mdl ]; then
    local/run_nnet.sh data/train_${trainid}
  else

    for t in data/dev_clean data/test_clean data/dev_clean_masked data/test_clean_masked; do
      if [ ! -f $exp/tri4_nnet_${trainid}/decode_${t##*/}/.done ]; then
        $train_cmd local/run_nnet.decode.sh $t $exp/tri3/graph $exp/tri4_nnet_${trainid} &
      fi
    done

    while read noise_type <&3; do
      while read snr <&4; do
        for t in data/dev_noisy_${noise_type}_snr_${snr}_masked data/test_noisy_${noise_type}_snr_${snr}_masked; do 
          #steps/compute_cmvn_stats.sh $t $exp/make_fbank_mfcc/`basename $t` fbank_mfcc
          #utils/fix_data_dir.sh $t
          if [ ! -f $exp/tri4_nnet_${trainid}/decode_${t##*/}/.done ]; then
            $train_cmd local/run_nnet.decode.sh $t $exp/tri3/graph $exp/tri4_nnet_${trainid} &
          fi
        done
      done 4< snr.list
    done 3< noisetypes.list
  fi
done
