. path.sh
. cmd.sh

set -e

datadir=data/train_noisy
nnet_dir=exp/irm_nnet
nj=4
cmd=run.pl 

. parse_options.sh

dirid=`basename $datadir`

sdata=$datadir/split$nj
utils/split_data.sh $datadir $nj

mkdir -p irm

$cmd JOB=1:$nj exp/make_irm/make_irm_$dirid.JOB.log \
  nnet2-compute --raw=true $nnet_dir/final.nnet scp:$sdata/JOB/feats.scp ark:- \| \
  irm-targets-to-irm --apply-log=true ark:- ark,scp:irm/irm_$dirid.JOB.ark,irm/irm_$dirid.JOB.scp

echo "$0: Creating masked Mel filterbank features in ${datadir}_masked..."

mkdir -p fbank

$cmd JOB=1:$nj exp/make_fbank/make_fbank_${dirid}_masked.JOB.log \
  matrix-sum scp:$sdata/JOB/feats.scp \
  scp:irm/irm_${dirid}.JOB.scp ark:- \| copy-feats ark:- \
  ark,scp:fbank/raw_fbank_${dirid}_masked.JOB.ark,fbank/raw_fbank_${dirid}_masked.JOB.scp

cp -rT $datadir ${datadir}_masked
rm ${datadir}_masked/{feats.scp,cmvn.scp}

for n in `seq $nj`; do 
  cat fbank/raw_fbank_${dirid}_masked.$n.scp
done | sort > ${datadir}_masked/feats.scp

steps/compute_cmvn_stats.sh --fake ${datadir}_masked exp/make_fbank fbank

echo "$0: Creating masked MFCC features in ${datadir}_masked_fbank_mfcc..."

cp -rT ${datadir}_masked ${datadir}_masked_fbank_mfcc
rm ${datadir}_masked_fbank_mfcc/{feats.scp,cmvn.scp}

sdata=${datadir}_masked/split$nj
utils/split_data.sh ${datadir}_masked $nj

dirid=${dirid}_masked

mkdir -p fbank_mfcc
mkdir -p exp/make_fbank_mfcc

utils/split_data.sh ${datadir}_fbank_mfcc $nj

$cmd JOB=1:$nj exp/make_fbank_mfcc/make_fbank_mfcc_${dirid}_masked.JOB.log \
  compute-mfcc-feats-from-fbank --config=conf/mfcc.conf \
  scp:$sdata/JOB/feats.scp \
  ark,scp:fbank_mfcc/raw_${dirid}_fbank_mfcc.JOB.ark,fbank_mfcc/raw_${dirid}_fbank_mfcc.JOB.scp || exit 1

for n in `seq $nj`; do 
  cat fbank_mfcc/raw_${dirid}_fbank_mfcc.$n.scp
done | sort > ${datadir}_masked_fbank_mfcc/feats.scp

steps/compute_cmvn_stats.sh ${datadir}_masked_fbank_mfcc exp/make_fbank_mfcc fbank_mfcc

echo "$0: Creating concat MFCC features in ${datadir}_concat_fbank_mfcc..."

cp -rT ${datadir}_fbank_mfcc ${datadir}_concat_fbank_mfcc
rm ${datadir}_concat_fbank_mfcc/{feats.scp,cmvn.scp}
rm -rf ${datadir}_concat_fbank_mfcc/split*

utils/split_data.sh ${datadir}_fbank_mfcc $nj
utils/split_data.sh ${datadir}_masked_fbank_mfcc $nj

$cmd JOB=1:$nj exp/make_fbank_mfcc/make_fbank_mfcc_${dirid}_concat.JOB.log \
  paste-feats scp:${datadir}_fbank_mfcc/split$nj/JOB/feats.scp \
  scp:${datadir}_masked_fbank_mfcc/split$nj/JOB/feats.scp \
  ark,scp:fbank_mfcc/raw_${dirid}_concat_fbank_mfcc.JOB.ark,fbank_mfcc/raw_${dirid}_concat_fbank_mfcc.JOB.scp || exit 1

for n in `seq $nj`; do 
  cat fbank_mfcc/raw_${dirid}_concat_fbank_mfcc.$n.scp
done | sort > ${datadir}_concat_fbank_mfcc/feats.scp

steps/compute_cmvn_stats.sh ${datadir}_concat_fbank_mfcc exp/make_fbank_mfcc fbank_mfcc

