#!/bin/bash

# Copyright 2014  Vimal Manohar (Johns Hopkins University)
# Apache 2.0

# Begin configuration
stage=-4      # This allows restarting after partway, when something when wrong.
cmd=run.pl
nj=4
target_dim=0  # If > 0, learn CCA of this dimension
splice_opts="--left-context=0 --right-context=0"
cmwn_opts="--norm-vars=false --norm-means=false"
regularizer_list="1e-4:1e-4"
add_deltas=false
# End configuration.

echo "$0 $@"  # Print the command line for logging

[ -f path.sh ] && . ./path.sh;
. parse_options.sh || exit 1;

if [ $# != 3 ]; then
  echo "Usage: local/train_cca.sh <data-dir-1> <data-dir-2> <exp-dir>"
  echo "e.g. : local/train_cca.sh data_clean/train_clean_multi data_noisy/train_multi exp/cca_clean_noisy"
  echo "main options (for others, see top of script file)"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --config <config-file>                           # config containing options"
  echo "  --stage <stage>                                  # stage to do partial re-run from."
  exit 1
fi

data_view1=$1
data_view2=$2
dir=$3

sdata_view1=$data_view1/split$nj
sdata_view2=$data_view2/split$nj

utils/split_data.sh $data_view1 $nj
utils/split_data.sh $data_view2 $nj

#feats_view1="ark,s,cs:apply-cmvn --utt2spk=ark:$data_view1/utt2spk scp:$data_view1/cmvn.scp scp:$data_view1/feats.scp ark:- |"
#feats_view2="ark,s,cs:apply-cmvn --utt2spk=ark:$data_view2/utt2spk scp:$data_view2/cmvn.scp scp:$data_view2/feats.scp ark:- |"

feats_view1="ark,s,cs:copy-feats scp:$sdata_view1/JOB/feats.scp ark:- |"
feats_view2="ark,s,cs:copy-feats scp:$sdata_view2/JOB/feats.scp ark:- |"

if $add_deltas; then
  feats_view1="$feats_view1 add-deltas ark:- ark:- |"
  feats_view2="$feats_view2 add-deltas ark:- ark:- |"
fi

feats_view1="$feats_view1 splice-feats $splice_opts ark:- ark:- |"
feats_view2="$feats_view2 splice-feats $splice_opts ark:- ark:- |"

mkdir -p $dir

echo $splice_opts > $dir/splice_opts
echo $add_deltas > $dir/add_delta

if [ $stage -le 1 ]; then
  $cmd --mem 4G JOB=1:$nj $dir/log/acc_cca_stats.JOB.log \
    acc-cca-stats "$feats_view1" "$feats_view2" $dir/cca_stats.JOB
fi
if [ $stage -le 2 ]; then
  $cmd --mem 4G $dir/log/combine_acc_stats.log \
    sum-cca-stats $dir/cca_stats.* $dir/cca_stats
fi
if [ $stage -le 3 ]; then
  if [ $target_dim -gt 0 ]; then
    $cmd --mem 4G $dir/log/est_cca.log \
      est-cca --regularizer-list="$regularizer_list" --target-dim=$target_dim \
      $dir/cca_stats $dir/trans12.mat $dir/trans21.mat \
      $dir/cmvn_global_view1 \
      $dir/cmvn_global_view2 || exit 1
  else
    $cmd --mem 4G $dir/log/est_cca.log \
      est-cca --regularizer-list="$regularizer_list" \
      $dir/cca_stats $dir/trans12.mat $dir/trans21.mat \
      $dir/cmvn_global_view1 \
      $dir/cmvn_global_view2 || exit 1
  fi
fi
