set -o pipefail

cmd=run.pl
testid=dev
regularization_list="1e-10:1e-10"
splice_opts="--left-context=0 --right-context=0"
target_dim=13

. path.sh
. utils/parse_options.sh

if [ $# -ne 4 ]; then 
  echo "Incorrect number of options: $#; Expecting 1"
  exit 1
fi

data_view1=$1
data_view2=$2
exp_dir=$3
dir=$4

mkdir -p $dir

while read noise_type <&3; do
  while read snr <&4; do
    x=${noise_type}_snr_$snr
    $cmd TARGET_DIM=1:$target_dim $dir/compute_correlation_${testid}_noisy_$x.TARGET_DIM.log \
      compute-correlation --target-dim=TARGET_DIM --regularizer-list="$regularization_list" \
      "ark:splice-feats $splice_opts scp:$data_view1/${testid}_clean_noisy_$x/feats.scp ark:- | apply-cmvn $exp_dir/cmvn_global_view1 ark:- ark:- | transform-feats $exp_dir/trans12.mat ark:- ark:- |" \
      "ark:splice-feats $splice_opts scp:$data_view2/${testid}_noisy_$x/feats.scp ark:- | apply-cmvn $exp_dir/cmvn_global_view1 ark:- ark:- | transform-feats $exp_dir/trans12.mat ark:- ark:- |" || exit 1
  done 4< conf/snr.list
done 3< conf/noisetypes.list
