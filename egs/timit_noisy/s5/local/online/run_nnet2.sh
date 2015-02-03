#!/bin/bash

. cmd.sh

stage=1
train_stage=-10
use_gpu=true
mix_up=5000
initial_learning_rate=0.002
final_learning_rate=0.0002
num_hidden_layers=3
pnorm_input_dim=2000
pnorm_output_dim=200
num_epochs=15
do_decode=false
egs_dir=
test=data_noisy/dev_multi

. cmd.sh
. ./path.sh
. ./utils/parse_options.sh

if [ $# -ne 4 ]; then
  echo "Usage: local/online/run_nnet2.sh <data-dir> <lang> <ali-dir> <exp-dir>"
  echo "e.g. : local/online/run_nnet2.sh data/train_clean data/lang exp/tri3_ali exp/tri4_nnet"
  exit 1
fi

train=$1
lang=$2
ali_dir=$3
dir=$4

if $use_gpu; then
  if ! cuda-compiled; then
    cat <<EOF && exit 1 
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA 
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.  Otherwise, call this script with --use-gpu false
EOF
  fi
  parallel_opts="-l gpu=1" 
  num_threads=1
  minibatch_size=512
else
  # Use 4 nnet jobs just like run_4d_gpu.sh so the results should be
  # almost the same, but this may be a little bit slow.
  num_threads=16
  minibatch_size=128
  parallel_opts="-pe smp $num_threads" 
fi

# stages 1 through 3 run in run_nnet2_common.sh.
# local/online/run_nnet2_common.sh --stage $stage || exit 1;

if [ $stage -le 4 ]; then
  if [[ `hostname -f` == "*.clsp.jhu.edu" ]]; then
    utils/create_split_dir.pl /export/b0{1,2,3,4}/$USER/kaldi-data/egs/timit_noisy_s5/$dir-$(date +'%m_%d_%H_%M')/egs $dir/egs/storage
  fi

  steps/nnet2/train_pnorm_simple2.sh --stage $train_stage \
    --splice-width 7 \
    --feat-type raw \
    --online-ivector-dir "" \
    --cmvn-opts "--norm-means=false --norm-vars=false" \
    --num-threads "$num_threads" \
    --minibatch-size "$minibatch_size" \
    --parallel-opts "$parallel_opts" \
    --num-jobs-nnet 4 \
    --num-epochs $num_epochs \
    --add-layers-period 1 \
    --num-hidden-layers $num_hidden_layers \
    --mix-up $mix_up \
    --initial-learning-rate $initial_learning_rate \
    --final-learning-rate $final_learning_rate \
    --cmd "$decode_cmd" \
    --pnorm-input-dim $pnorm_input_dim \
    --pnorm-output-dim $pnorm_output_dim \
    --egs-dir "$egs_dir" \
    $train $lang $ali_dir $dir || exit 1;
fi

if ! $do_decode; then
  echo "Done training neural network. But not running decode."
  exit 0
fi

#if [ $stage -le 5 ]; then
#  steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" \
#    --nj 4 data/test exp/nnet2_online/extractor \
#    exp/nnet2_online/ivectors_test || exit 1;
#fi

if [ $stage -le 6 ]; then
  # Note: comparing the results of this with run_online_decoding_nnet2_baseline.sh,
  # it's a bit worse, meaning the iVectors seem to hurt at this amount of data.
  # However, experiments by Haihua Xu (not checked in yet) on WSJ, show it helping
  # nicely.  This setup seems to have too little data for it to work, but it suffices
  # to demonstrate the scripts.   We will likely modify it to add noise to the
  # iVectors in training, which will tend to mitigate the over-training.
  steps/nnet2/decode.sh --cmd "$decode_cmd" --nj 10 \
    exp_clean/tri3/graph $test $dir/decode_`basename $test`  &

  wait
fi

exit 0

if [ $stage -le 7 ]; then
  # If this setup used PLP features, we'd have to give the option --feature-type plp
  # to the script below.
  steps/online/nnet2/prepare_online_decoding.sh data/lang exp/nnet2_online/extractor \
    "$dir" ${dir}_online || exit 1;
fi

if [ $stage -le 8 ]; then
  # do the actual online decoding with iVectors.
  steps/online/nnet2/decode.sh --config conf/decode.config --cmd "$decode_cmd" --nj 20 \
    exp/tri3b/graph data/test ${dir}_online/decode &
  wait
fi

if [ $stage -le 9 ]; then
  # this version of the decoding treats each utterance separately
  # without carrying forward speaker information.
  steps/online/nnet2/decode.sh --config conf/decode.config --cmd "$decode_cmd" --nj 20 \
    --per-utt true \
    exp/tri3b/graph data/test ${dir}_online/decode_per_utt &
  wait
fi

exit 0;
