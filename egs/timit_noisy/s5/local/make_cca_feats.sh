#!/bin/bash

# Copyright 2014  Vimal Manohar (Johns Hopkins University)
# Apache 2.0

# Begin configuartion section.
nj=4
cmd=run.pl
target_dim=0
view1=false
# End configuration section.

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# -ne 5 ]; then 
  echo "Usage: $0 <in-data-dir> <exp-dir> <log-dir> <path-to-featdir> <output-data-dir>";
  echo "e.g. : $0 data/dev exp/cca_mfcc exp/make_cca_feats cca_feats data_cca/dev"
  echo "options: "
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

trans_mat=$srcdir/trans21.mat
cmvn_global=$srcdir/cmvn_global_view2

if $view1; then 
  trans_mat=$srcdir/trans12.mat
  cmvn_global=$srcdir/cmvn_global_view1
fi

extra_files=

for f in $data/feats.scp $trans_mat $cmvn_global $extra_files; do
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done

mkdir -p $tempdir || exit 1
mkdir -p $dir || exit 1
mkdir -p $featdir || exit 1

sdata=$data/split$nj
utils/split_data.sh $data $nj

select_dims="-"
if [ $target_dim -gt 0 ]; then
  select_dims=0-$[target_dim-1]
fi

$cmd JOB=1:$nj $tempdir/log/make_cca_feats.JOB.log \
  apply-cmvn $cmvn_global scp:$sdata/JOB/feats.scp ark:- \| \
  transform-feats $trans_mat ark:- ark:- \| \
  select-feats $select_dims ark:- \| \
  ark,scp:$featdir/raw_cca_feats_$name.JOB.ark,$featdir/raw_cca_feats_$name.JOB.scp || exit 1

for n in `seq $nj`; do
  cat $featdir/raw_cca_feats_$name.$n.scp || exit 1
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

echo "Succeeded creating cca features for $name"
