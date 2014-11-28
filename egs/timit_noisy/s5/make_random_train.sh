function make_random_train_dir {
  in_data=$1
  out_data=$2

  cp -rT $in_data $out_data

  utils/filter_scp.pl data/local/random_train/uttlist.multi $in_data/feats.scp > $out_data/feats.scp
  utils/fix_data_dir.sh $out_data

  for f in wav.scp feats.scp cmvn.scp utt2spk text; do
    sed 's:[^ ]*-snr-[0-9]*-::g' $out_data/$f | sort -k1,1 > $out_data/$f.tmp
    mv $out_data/$f.tmp $out_data/$f
  done

  utils/utt2spk_to_spk2utt.pl $out_data/utt2spk > $out_data/spk2utt
  utils/fix_data_dir.sh $out_data
}

if [ ! -f data/local/random_train/uttlist.multi ]; then
  mkdir -p data/local/random_train
  cut -d ' ' -f 1 data/train_clean/feats.scp > data/local/random_train/uttlist.clean

  declare -a rand_noise

  rand_noise[0]=""

  i=1
  for noise in `cat noisetypes.list`; do
    for snr in `cat snr.list`; do
      rand_noise[i]="$noise-snr-$snr-"
      i=$((i+1))
    done
  done


  for x in `cat data/local/random_train/uttlist.clean`; do
    echo "${rand_noise[$(($RANDOM % $i))]}$x"
  done | sort > data/local/random_train/uttlist.multi
fi

#make_random_train_dir data/train_multi data/train_random
#make_random_train_dir data/train_multi_masked data/train_random_masked
#make_random_train_dir data/train_multi_concat data/train_random_concat


