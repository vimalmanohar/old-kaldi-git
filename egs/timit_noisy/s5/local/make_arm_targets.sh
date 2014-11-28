#!/bin/bash

# Copyright 2014  Vimal Manohar

. ./cmd.sh 
[ -f path.sh ] && . ./path.sh
set -e

nj=30
cmd=run.pl
stage=-1
config=conf/arm.conf

. parse_options.sh

echo $*

if [ $# -ne 4 ]; then
  echo "$0: Incorrect number of arguments."
  echo "Usage:"
  echo "local/timit_prepare_arm_data.sh <noisy_datadir> <clean_datadir> <temp_dir>"
  echo "e.g.: local/timit_prepare_arm_data.sh data_fbank/train_noisy_babble_snr_10 data_fbank/train_clean exp/make_arm_targets arm_targets"
  echo 1
fi

datadir=$1
clean_datadir=$2
tmpdir=$3
dir=$4

# make $dir an absolute pathname.
dir=`perl -e '($dir,$pwd)= @ARGV; if($dir!~m:^/:) { $dir = "$pwd/$dir"; } print $dir; ' $dir ${PWD}`

dirid=`basename $datadir`

required="$datadir/feats.scp $clean_datadir/feats.scp $config"

for f in $required; do
  if [ ! -f $f ]; then
    echo "make_arm_targets.sh: no such file $f"
    exit 1;
  fi
done

utils/split_data.sh $datadir $nj
utils/split_data.sh $clean_datadir $nj

echo "$0: Compute arm targets using clean and noisy Mel filterbank features for $datadir..."

mkdir -p $tmpdir || exit 1

if [ $stage -le 2 ]; then
  $cmd JOB=1:$nj $tmpdir/arm_targets.${dirid}.JOB.log \
    compute-arm-targets --config=$config scp:$clean_datadir/split$nj/JOB/feats.scp \
    scp:$datadir/split$nj/JOB/feats.scp \
    ark,scp:$dir/${dirid}_arm_targets.JOB.ark,$dir/${dirid}_arm_targets.JOB.scp || exit 1

  for n in `seq $nj`; do 
    cat $dir/${dirid}_arm_targets.$n.scp
  done | sort -k1,1 > $datadir/arm_targets.scp
fi

