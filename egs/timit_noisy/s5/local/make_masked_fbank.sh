. path.sh
. cmd.sh

set -e

nj=4
cmd=run.pl 
stage=-1
config=
mask_type=irm

. parse_options.sh

if [ $# -ne 6 ]; then
  echo "$0: Incorrect number of arguments"
  echo "Usage:"
  echo "local/make_masked_fbank.sh <in-data-dir> <out-data-dir> <irm-nnet-dir> <log-dir> <irm-dir> <feat-dir>"
  echo "e.g.: local/make_masked_fbank.sh data_noisy_fbank/train_multi data_noisy_fbank/train_multi_masked exp/irm_nnet exp/make_masked_fbank/train_multi masked_fbank"
fi

in_datadir=$1
out_datadir=$2
nnet_dir=$3
logdir=$4
irm_dir=$5
dir=$6

if [ -z "$config" ]; then
  config=conf/${mask_type}.conf
fi

if [ $mask_type != "irm" ] && [ $mask_type != "arm" ]; then
  echo "Unknown mask type: $mask_type. Expecting irm or arm"
  exit 1
fi

# make $dir an absolute pathname.
dir=`perl -e '($dir,$pwd)= @ARGV; if($dir!~m:^/:) { $dir = "$pwd/$dir"; } print $dir; ' $dir ${PWD}`

# make $irm_dir an absolute pathname.
irm_dir=`perl -e '($dir,$pwd)= @ARGV; if($dir!~m:^/:) { $dir = "$pwd/$dir"; } print $dir; ' $irm_dir ${PWD}`


dirid=`basename $in_datadir`

required="$config $in_datadir/feats.scp $nnet_dir/final.nnet"
for f in $required; do
  if [ ! -f $f ]; then
    echo "make_masked_fbank.sh: no such file $f"
    exit 1;
  fi
done

sdata=$in_datadir/split$nj
utils/split_data.sh $in_datadir $nj

mkdir -p $irm_dir

if [ $stage -le 0 ]; then
  if [ $mask_type == "irm" ]; then
    $cmd JOB=1:$nj $logdir/make_irm_$dirid.JOB.log \
      nnet2-compute --raw=true $nnet_dir/final.nnet scp:$sdata/JOB/feats.scp ark:- \| \
      irm-targets-to-irm --apply-log=true --config=$config ark:- ark,scp:$irm_dir/irm_$dirid.JOB.ark,$irm_dir/irm_$dirid.JOB.scp
  else
    $cmd JOB=1:$nj $logdir/make_arm_$dirid.JOB.log \
      nnet2-compute --raw=true $nnet_dir/final.nnet scp:$sdata/JOB/feats.scp ark:- \| \
      arm-targets-to-arm --apply-log=true --config=$config ark:- ark,scp:$irm_dir/arm_$dirid.JOB.ark,$irm_dir/arm_$dirid.JOB.scp

  fi
fi

echo "$0: Creating masked Mel filterbank features in $out_datadir"

mkdir -p $dir

if [ $stage -le 1 ]; then
  if [ $mask_type == "irm" ]; then
    $cmd JOB=1:$nj $logdir/make_fbank_${dirid}_masked.JOB.log \
      matrix-sum scp:$sdata/JOB/feats.scp \
      scp:$irm_dir/irm_${dirid}.JOB.scp ark:- \| copy-feats ark:- \
      ark,scp:$dir/raw_fbank_${dirid}_masked.JOB.ark,$dir/raw_fbank_${dirid}_masked.JOB.scp
  else
    $cmd JOB=1:$nj $logdir/make_fbank_${dirid}_masked.JOB.log \
      matrix-sum scp:$sdata/JOB/feats.scp \
      scp:$irm_dir/arm_${dirid}.JOB.scp ark:- \| copy-feats ark:- \
      ark,scp:$dir/raw_fbank_${dirid}_masked.JOB.ark,$dir/raw_fbank_${dirid}_masked.JOB.scp
  fi
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
  cat $irm_dir/arm_$dirid.$n.scp
done | sort -k1,1 > ${in_datadir}/arm.scp

echo "$0: Succeeded creating masked Mel filterbank features in $out_datadir"
