#!/bin/bash

# Copyright 2014  Vimal Manohar
# Apache 2.0

set -e

# Begin configuration section.
cmd=run.pl
stage=0
nj=4
transform_dir=
raw=false         # Use raw nnet directly
iter=final

# End configuration options.

echo "$0 $@"  # Print the command line for logging

[ -f path.sh ] && . ./path.sh # source the path.
. parse_options.sh || exit 1;

if [ $# != 5 ]; then
   echo "usage: steps/nnet2/dump_posteriors.sh <input-data-dir> <nnet-dir> <log-dir> <feat-dir> <output-data-dir>"
   echo "e.g.:  steps/nnet2/dump_posteriors.sh data/train exp/nnet_multiview exp/make_dcca_feat dcca_feat data_dcca/train"
   echo "main options (for others, see top of script file)"
   echo "  --config <config-file>                           # config containing options"
   echo "  --nj <nj>                                        # number of parallel jobs"
   echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
   exit 1;
fi

data=$1
srcdir=$2
tempdir=$3
featdir=$4
dir=$5

name=`basename $data`
tempdir=$tempdir/$name

featdir=`perl -e '($dir,$pwd)= @ARGV; if($dir!~m:^/:) { $dir = "$pwd/$dir"; } print $dir; ' $featdir ${PWD}`

if $raw; then
  model=$srcdir/$iter.nnet
  nnet=$model
else 
  model=$srcdir/$iter.mdl
  nnet="nnet-to-raw-nnet $model - |"
fi

[ ! -z "$online_ivector_dir" ] && \
  extra_files="$online_ivector_dir/ivector_online.scp $online_ivector_dir/ivector_period"

for f in $data/feats.scp $model $extra_files; do
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done

mkdir -p $tempdir
mkdir -p $dir
mkdir -p $featdir

# because we [cat trans.*], no need to keep nj consistent with [# of trans]
[ ! -z "$transform_dir" ] && nj=`cat $transform_dir/num_jobs`
echo $nj > $tempdir/num_jobs

cmvn_opts=`cat $srcdir/cmvn_opts 2>/dev/null`
sdata=$data/split$nj
utils/split_data.sh $data $nj

if [ -f $srcdir/final.mat ] && [ ! -f $transform_dir/raw_trans.1 ]; then
  feat_type=lda; else feat_type=raw; fi
echo "$0: feature type is $feat_type"

case $feat_type in
  raw) feats="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- |"
    echo $cmvn_opts > $tempdir/cmvn_opts
    ;;
  lda)
    splice_opts=`cat $srcdir/splice_opts 2>/dev/null`
    cp $srcdir/{splice_opts,cmvn_opts,final.mat} $tempdir || exit 1;
    [ ! -z "$cmvn_opts" ] && \
       echo "You cannot supply --cmvn-opts option if feature type is LDA." && exit 1;
    cmvn_opts=$(cat $tempdir/cmvn_opts)

    feats="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | splice-feats $splice_opts ark:- ark:- | transform-feats $tempdir/final.mat ark:- ark:- |"
    ;;
  *) echo "$0: invalid feature type $feat_type" && exit 1;
esac

if [ -f $transform_dir/trans.1 ] && [ $feat_type != "raw" ]; then
  echo "$0: using transforms from $transform_dir"
  feats="$feats transform-feats --utt2spk=ark:$sdata/JOB/utt2spk ark:$transform_dir/trans.JOB ark:- ark:- |"
fi

if [ -f $transform_dir/raw_trans.1 ] && [ $feat_type == "raw" ]; then
  echo "$0: using raw-fMLLR transforms from $transform_dir"
  feats="$feats transform-feats --utt2spk=ark:$sdata/JOB/utt2spk ark:$transform_dir/raw_trans.JOB ark:- ark:- |"
fi

if [ ! -z "$online_ivector_dir" ]; then
  ivector_period=$(cat $online_ivector_dir/ivector_period) || exit 1;
  # note: subsample-feats, with negative n, will repeat each feature -n times.
  feats="$feats paste-feats --length-tolerance=$ivector_period ark:- 'ark,s,cs:utils/filter_scp.pl $sdata/JOB/utt2spk $online_ivector_dir/ivector_online.scp | subsample-feats --n=-$ivector_period scp:- ark:- |' ark:- |"
fi

## Set up input features of nnet
if [ -z "$feat_type" ]; then
  if [ -f $nnetdir/final.mat ]; then feat_type=lda; else feat_type=delta; fi
fi
echo "$0: feature type is $feat_type"

case $feat_type in
  raw) feats="ark,s,cs:apply-cmvn --norm-vars=false --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- |";;
  delta) feats="ark,s,cs:apply-cmvn --norm-vars=false --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | add-deltas ark:- ark:- |";;
  lda) feats="ark,s,cs:apply-cmvn --norm-vars=false --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | splice-feats $splice_opts ark:- ark:- | transform-feats $nnetdir/final.mat ark:- ark:- |"
   ;;
  *) echo "Invalid feature type $feat_type" && exit 1;
esac

if [ ! -z "$transform_dir" ]; then
  echo "Using transforms from $transform_dir"
  [ ! -f $transform_dir/trans.1 ] && echo "No such file $transform_dir/trans.1" && exit 1;
#  cat $transform_dir/trans.* > $nnetdir/trans || exit 1;
  feats="$feats transform-feats --utt2spk=ark:$sdata/JOB/utt2spk ark,s,cs:$transform_dir/trans.JOB ark:- ark:- |"
fi

if [ $stage -le 1 ]; then
  echo "Making posterior features scp and ark."
  $cmd JOB=1:$nj $tempdir/make_post.JOB.log \
    nnet-compute "$nnet" "$feats" ark:- \| \
    copy-feats --compress=true ark:- ark,scp:$featdir/raw_post_feat_$name.JOB.ark,$featdir/raw_post_feat_$name.JOB.scp || exit 1;
fi

for n in $(seq $nj); do
  cat $featdir/raw_post_feat_$name.$n.scp || exit 1;
done > $dir/feats.scp

for f in segments spk2utt text utt2spk wav.scp char.stm glm kws reco2file_and_channel stm utt2uniq spk2uniq; do
  [ -e $data/$f ] && cp $data/$f $dir/$f
done

nf=`cat $dir/feats.scp | wc -l` 
nu=`cat $dir/utt2spk | wc -l` 
if [ $nf -ne $nu ]; then
  echo "It seems not all of the feature files were successfully ($nf != $nu);"
  echo "consider using utils/fix_data_dir.sh $dir"
fi

echo "Succeeded creating posterior features for $name"
