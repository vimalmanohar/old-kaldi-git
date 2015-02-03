nj=10
cmd=run.pl
target_dim=13

. path.sh
. utils/parse_options.sh

if [ $# -ne 6 ]; then
  exit 1
fi

mfcc_dir=$1
cca_dir=$2
cca_model=$3
dir=$4
tmp_dir=$5
featdir=$6

mkdir -p $dir $tmp_dir/cca_dim$target_dim $featdir

utils/split_data.sh $mfcc_dir $nj || exit 1
utils/split_data.sh $cca_dir $nj || exit 1

cp $mfcc_dir/* $tmp_dir/cca_dim$target_dim

$cmd JOB=1:$nj $tmp_dir/cca_dim$target_dim/log/select_feats.JOB.log select-feats 0-$[target_dim-1] "ark:apply-cmvn $cca_model/cmvn_global_view2 scp:$cca_dir/split$nj/JOB/feats.scp ark:- | transform-feats $cca_model/trans21.mat ark:- ark:- |" ark,scp:$tmp_dir/cca_dim$target_dim/raw_cca_feats.JOB.ark,$tmp_dir/cca_dim$target_dim/raw_cca_feats.JOB.scp || exit 1

for n in `seq $nj`; do
  cat $tmp_dir/cca_dim$target_dim/raw_cca_feats.$n.scp
done > $tmp_dir/cca_dim$target_dim/feats.scp

steps/append_feats.sh --cmd "$cmd" --nj $nj $mfcc_dir $tmp_dir/cca_dim$target_dim $dir $tmp_dir $featdir || exit 1
steps/compute_cmvn_stats.sh $dir $tmp_dir $featdir || exit 1
