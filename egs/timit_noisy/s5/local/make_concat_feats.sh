#!/bin/bash 

# Copyright 2014  Vimal Manohar
# Apache 2.0
# To be run from .. (one directory up from here)
# see ../run.sh for example

# Begin configuration section.
nj=4
cmd=run.pl
compress=true
# End configuration section.

set -e
set -o pipefail
set -u

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# -lt 4 ]; then
   echo "usage: make_concat_feats.sh [options] <out-data-dir> <data-dir1> <data-dir2> ... <data-dirn> <log-dir> <path-to-featdir>";
   echo "options: "
   echo "  --nj <nj>                                        # number of parallel jobs"
   echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
   exit 1;
fi

out_datadir=$1
logdir=${@: -2:1}
mfccdir=${@: -1}
shift 1;

datadirs=( $@ )
unset datadirs[${#datadirs[@]}-1]
unset datadirs[${#datadirs[@]}-1]

# make $mfccdir an absolute pathname.
mfccdir=`perl -e '($dir,$pwd)= @ARGV; if($dir!~m:^/:) { $dir = "$pwd/$dir"; } print $dir; ' $mfccdir ${PWD}`

# use "name" as part of name of the archive.
name=`basename $out_datadir`

mkdir -p $mfccdir || exit 1;
mkdir -p $logdir || exit 1;

nu=`cat ${datadirs[0]}/feats.scp | wc -l`

feats=
for d in `echo ${datadirs[@]} | tr ' ' '\n'`; do
  if [ ! -f $d/feats.scp ]; then
    echo "$0: no such file $d/feats.scp"
    exit 1;
  fi
  
  if [ "`cat $d/feats.scp | wc -l`" -ne "$nu" ]; then
    echo "$0: WARNING: number of utterances in ${datadirs[0]} and $d are different"
  fi

  utils/validate_data_dir.sh --no-text $d || exit 1;
  utils/split_data.sh $d $nj

  feats="$feats scp:$d/split$nj/JOB/feats.scp"
done

cp -rT ${datadirs[0]} $out_datadir
set +e
rm $out_datadir/{feats.scp,cmvn.scp}
rm -rf $out_datadir/split*
set -e

$cmd JOB=1:$nj $logdir/make_feats_${name}.JOB.log \
  paste-feats$feats ark,scp:$mfccdir/raw_feats_${name}.JOB.ark,$mfccdir/raw_feats_${name}.JOB.scp || exit 1

if [ -f $logdir/.error.$name ]; then
  echo "Error producing concatenated features for $name:"
  tail $logdir/make_feats_${name}.1.log
  exit 1;
fi

# concatenate the .scp files together.
for ((n=1; n<=nj; n++)); do
  cat $mfccdir/raw_feats_$name.$n.scp || exit 1;
done > $out_datadir/feats.scp

nf=`cat $out_datadir/feats.scp | wc -l` 
nu=`cat $out_datadir/utt2spk | wc -l` 
if [ $nf -ne $nu ]; then
  echo "It seems not all of the feature files were successfully processed ($nf != $nu);"
  echo "consider using utils/fix_data_dir.sh $out_datadir"
fi

if [ $nf -lt $[$nu - ($nu/20)] ]; then
  echo "Less than 95% the features were successfully generated.  Probably a serious error."
  exit 1;
fi

echo "Succeeded creating concatenated features for $name"

