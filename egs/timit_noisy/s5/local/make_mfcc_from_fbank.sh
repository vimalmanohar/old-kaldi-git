#!/bin/bash 

# Copyright 2014  Vimal Manohar
# Apache 2.0
# To be run from .. (one directory up from here)
# see ../run.sh for example

# Begin configuration section.
nj=4
cmd=run.pl
mfcc_config=conf/mfcc.conf
compress=true
# End configuration section.

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# != 4 ]; then
   echo "usage: make_mfcc_from_bank.sh [options] <fbank-data-dir> <data-dir> <log-dir> <path-to-mfccdir>";
   echo "options: "
   echo "  --mfcc-config <config-file>                      # config passed to compute-mfcc-feats "
   echo "  --nj <nj>                                        # number of parallel jobs"
   echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
   exit 1;
fi

fbank_data=$1
data=$2
logdir=$3
mfccdir=$4

# make $mfccdir an absolute pathname.
mfccdir=`perl -e '($dir,$pwd)= @ARGV; if($dir!~m:^/:) { $dir = "$pwd/$dir"; } print $dir; ' $mfccdir ${PWD}`

# use "name" as part of name of the archive.
name=`basename $data`

mkdir -p $mfccdir || exit 1;
mkdir -p $logdir || exit 1;

scp=$fbank_data/feats.scp
required="$scp $mfcc_config"

for f in $required; do
  if [ ! -f $f ]; then
    echo "make_mfcc.sh: no such file $f"
    exit 1;
  fi
done
utils/validate_data_dir.sh --no-text $fbank_data || exit 1;

cp -rT $fbank_data $data
set +e
rm $data/{feats.scp,cmvn.scp}
rm -rf $data/split*
set -e

utils/split_data.sh ${fbank_data} $nj
sdata=$fbank_data/split$nj

$cmd JOB=1:$nj $logdir/make_fbank_mfcc_${name}.JOB.log \
  compute-mfcc-feats-from-fbank --config=$mfcc_config \
  scp:$sdata/JOB/feats.scp \
  ark,scp:$mfccdir/raw_fbank_mfcc_${name}.JOB.ark,$mfccdir/raw_fbank_mfcc_${name}.JOB.scp || exit 1

if [ -f $logdir/.error.$name ]; then
  echo "Error producing mfcc features for $name from fbank features:"
  tail $logdir/make_fbank_mfcc_${name}.1.log
  exit 1;
fi

# concatenate the .scp files together.
for ((n=1; n<=nj; n++)); do
  cat $mfccdir/raw_fbank_mfcc_$name.$n.scp || exit 1;
done > $data/feats.scp

nf=`cat $data/feats.scp | wc -l` 
nu=`cat $data/utt2spk | wc -l` 
if [ $nf -ne $nu ]; then
  echo "It seems not all of the feature files were successfully processed ($nf != $nu);"
  echo "consider using utils/fix_data_dir.sh $data"
fi

if [ $nf -lt $[$nu - ($nu/20)] ]; then
  echo "Less than 95% the features were successfully generated.  Probably a serious error."
  exit 1;
fi

echo "Succeeded creating MFCC features for $name from fbank features"

