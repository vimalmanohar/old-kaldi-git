function make_fbank {
  src_dir=$1
  dest_dir=$2
  fbankdir=$3

  [ -z "$nj" ] && echo "\$nj is not set!" && exit 1

  y=${dest_dir##*/}

  cp -rT $src_dir $dest_dir
  set +e
  rm $dest_dir/{feats.scp,cmvn.scp}
  rm -rf $dest_dir/split*
  set -e
  steps/make_fbank.sh --cmd "$train_cmd" --nj $nj $dest_dir $exp/make_fbank/$y $fbankdir
  steps/compute_cmvn_stats.sh --fake $dest_dir $exp/make_fbank/$y $fbankdir
  utils/fix_data_dir.sh $dest_dir
}

function make_fbank_mfcc {
  src_dir=$1
  dest_dir=$2
  mfccdir=$3

  [ -z "$nj" ] && echo "\$nj is not set!" && exit 1

  y=${dest_dir##*/}
    
  local/make_mfcc_from_fbank.sh --cmd "$train_cmd" --nj $nj $src_dir $dest_dir $exp/make_fbank_mfcc/$y $mfccdir
  steps/compute_cmvn_stats.sh $dest_dir $exp/make_fbank_mfcc/$y $mfccdir
  utils/fix_data_dir.sh $dest_dir
}

function make_masked_fbank {
  src_dir=$1
  dest_dir=$2
  irm_nnet_dir=$3
  irm_dir=$4
  fbankdir=$5

  y=${dest_dir##*/}

  local/make_masked_fbank.sh --cmd "$train_cmd" --nj $nj \
    $src_dir $dest_dir $irm_nnet_dir $exp/make_masked_fbank/$y $irm_dir $fbankdir
  steps/compute_cmvn_stats.sh --fake $dest_dir $exp/make_masked_fbank/$y $fbankdir
  utils/fix_data_dir.sh $dest_dir
}

function make_concat_mfcc {
  dest_dir=$1
  feat1_dir=$2
  feat2_dir=$3
  mfccdir=$4

  y=${dest_dir##*/}

  local/make_concat_feats.sh --cmd "$train_cmd" --nj $nj \
    $dest_dir $feat1_dir $feat2_dir $exp/make_fbank_mfcc/$y $mfccdir
  steps/compute_cmvn_stats.sh $dest_dir $exp/make_fbank_mfcc/$y $mfccdir
  utils/fix_data_dir.sh $dest_dir
}
