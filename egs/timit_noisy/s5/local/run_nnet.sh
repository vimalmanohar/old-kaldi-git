. path.sh
. cmd.sh

train_stage=-100
transform_dir=exp/tri3_ali
mixup=5000
initial_learning_rate=0.008
final_learning_rate=0.0008
num_hidden_layers=3
pnorm_input_dim=2000
pnorm_output_dim=200
num_epochs=15
num_epochs_extra=5

# DNN hybrid system training parameters
dnn_gpu_parallel_opts=(--minibatch-size 512 --num-jobs-nnet 4 --num-threads 1 \
  --parallel-opts "-l gpu=1")

. parse_options.sh

if [ $# -ne 4 ]; then
  echo "Usage: local/run_nnet.sh <data-dir> <lang> <ali-dir> <exp-dir>"
  echo "e.g. : local/run_nnet.sh data/train_clean data/lang exp/tri3_ali exp/tri4_nnet"
  exit 1
fi

train=$1
lang=$2
ali_dir=$3
exp_dir=$4

trainid=`basename $train`

set -e
set -o pipefail

echo ============================================================================
echo "                    DNN Hybrid Training & Decoding                        "
echo ============================================================================

if [[ `hostname -f` == "*.clsp.jhu.edu" ]]; then
  utils/create_split_dir.pl /export/b0{1,2,3,4}/$USER/kaldi-data/egs/timit_noisy_s5/$exp_dir/egs $exp_dir/egs/storage
fi

steps/nnet2/train_pnorm_fast.sh \
  --num-epochs $num_epochs --num-epochs-extra $num_epochs_extra \
  --initial-learning-rate $initial_learning_rate \
  --final-learning-rate $final_learning_rate \
  --num-hidden-layers $num_hidden_layers \
  --pnorm-input-dim $pnorm_input_dim \
  --pnorm-output-dim $pnorm_output_dim \
  --mix-up $mixup \
  --num-jobs-nnet 4 --cmd "$train_cmd" \
  "${dnn_gpu_parallel_opts[@]}" \
  --stage $train_stage \
  --transform-dir "$transform_dir" \
  $train $lang $ali_dir $exp_dir
