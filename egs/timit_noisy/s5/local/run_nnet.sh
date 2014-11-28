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

# DNN hybrid system training parameters
dnn_gpu_parallel_opts=(--minibatch-size 512 --max-change 40 --num-jobs-nnet 4 --num-threads 1 \
  --parallel-opts "-l gpu=1" --cmd "queue.pl -l arch=*64 -l mem_free=2G,ram_free=1G")

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

steps/nnet2/train_pnorm.sh --mix-up $mixup \
  --initial-learning-rate $initial_learning_rate \
  --final-learning-rate $final_learning_rate \
  --num-hidden-layers $num_hidden_layers \
  --pnorm-input-dim $pnorm_input_dim \
  --pnorm-output-dim $pnorm_output_dim \
  --num-jobs-nnet "$train_nj" --cmd "$train_cmd" \
  "${dnn_gpu_parallel_opts[@]}" \
  --stage $train_stage \
  --transform-dir "$transform_dir" \
  $train $lang $ali_dir $exp_dir
