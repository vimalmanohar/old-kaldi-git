. path.sh
. cmd.sh

set -e

nj=4
cmd=run.pl 
stage=-1
irm_config=conf/irm.conf

. parse_options.sh

if [ $# -ne 5 ]; then
  echo "$0: Incorrect number of arguments"
  echo "Usage:"
  echo "local/make_cheat_masked_fbank.sh <in-data-dir> <out-data-dir> <log-dir> <irm-dir> <feat-dir>"
  echo "e.g.: local/make_masked_fbank.sh data_noisy_fbank/train_multi data_noisy_fbank/train_multi_masked exp/irm_nnet exp/make_masked_fbank/train_multi masked_fbank"
  exit 1
fi

in_datadir=$1
out_datadir=$2
logdir=$3
irm_dir=$4
dir=$5

# make $dir an absolute pathname.
dir=`perl -e '($dir,$pwd)= @ARGV; if($dir!~m:^/:) { $dir = "$pwd/$dir"; } print $dir; ' $dir ${PWD}`

# make $irm_dir an absolute pathname.
irm_dir=`perl -e '($dir,$pwd)= @ARGV; if($dir!~m:^/:) { $dir = "$pwd/$dir"; } print $dir; ' $irm_dir ${PWD}`


dirid=`basename $in_datadir`

required="$irm_config $in_datadir/feats.scp $in_datadir/irm_targets.scp"
for f in $required; do
  if [ ! -f $f ]; then
    echo "make_masked_fbank.sh: no such file $f"
    exit 1;
  fi
done

sdata=$in_datadir/split$nj
utils/split_data.sh $in_datadir $nj

utils/split_scp.pl $in_datadir/irm_targets.scp `for n in $(seq $nj); do echo $sdata/$n/irm_targets.scp; done | tr '\n' ' '`

mkdir -p $irm_dir

if [ $stage -le 0 ]; then
  $cmd JOB=1:$nj $logdir/make_irm_$dirid.JOB.log \
    irm-targets-to-irm --apply-log=true --config=$irm_config \
    scp:$sdata/JOB/irm_targets.scp ark,scp:$irm_dir/irm_$dirid.JOB.ark,$irm_dir/irm_$dirid.JOB.scp
fi

echo "$0: Creating masked Mel filterbank features in $out_datadir"

mkdir -p $dir

if [ $stage -le 1 ]; then
  $cmd JOB=1:$nj $logdir/make_fbank_${dirid}_masked.JOB.log \
    matrix-sum scp:$sdata/JOB/feats.scp \
    scp:$irm_dir/irm_${dirid}.JOB.scp ark:- \| copy-feats ark:- \
    ark,scp:$dir/raw_fbank_${dirid}_masked.JOB.ark,$dir/raw_fbank_${dirid}_masked.JOB.scp
fi

cp -rT $in_datadir ${out_datadir}
set +e
rm ${out_datadir}/{feats.scp,cmvn.scp}
rm -rf ${out_datadir}/split*
set -e

for n in `seq $nj`; do 
  cat $dir/raw_fbank_${dirid}_masked.$n.scp
done | sort -k1,1 > ${out_datadir}/feats.scp

for n in `seq $nj`; do 
  cat $irm_dir/irm_$dirid.$n.scp
done | sort -k1,1 > ${in_datadir}/irm.scp

echo "$0: Succeeded creating masked Mel filterbank features in $out_datadir"
