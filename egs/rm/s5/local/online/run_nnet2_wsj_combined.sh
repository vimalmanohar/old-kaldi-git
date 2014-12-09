#!/bin/bash

# This script assumes you have previously run the WSJ example script including
# the optional part local/online/run_online_decoding_nnet2.sh.  It builds a
# neural net for online decoding on top of the network we previously trained on
# WSJ, by keeping everything but the last layer of that network and then
# training just the last layer on our data.  We then train the whole thing.

set -e

stage=0
train_stage=-10
use_gpu=true
dir=exp/nnet2_online/multilingual_wsj_c
src_dir=../../wsj/s5/exp/nnet2_online/nnet_a_gpu

. cmd.sh
. ./path.sh
. ./utils/parse_options.sh

# rm alignments:
# rm egs:
# wsj alignments:
# wsj egs:
# model
#
# bolt alignments: exp/tri5b_ali
# bolt egs: exp/nnet2_online_gale/nnet_ms_b_combined/egs
# gale alignments: ../../gale_arabic/s5/exp/tri3b
# gale egs ../../gale_arabic/s5/exp/nnet2_online_pitch_8khz/nnet_a/egs/
# model ../../gale_arabic/s5/exp/nnet2_online_pitch_8khz/nnet_a/10.mdl
# 

if $use_gpu; then
  if ! cuda-compiled; then
    cat <<EOF && exit 1 
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA 
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.  Otherwise, call this script with --use-gpu false
EOF
  fi
  num_threads=1
  num_gpu=1
  minibatch_size=512
  dir=${dir}_gpu
else
  num_threads=16
  num_gpu=0
  minibatch_size=128
fi

if [ $stage -le 0 ]; then
  # This version of the get_egs.sh script does the feature extraction and iVector
  # extraction in a single binary, reading the config, as part of the script.

  if [[ `hostname -f` == *.clsp.jhu.edu ]] && [ ! -d ${dir}_online/wsj_egs/storage ]; then
    utils/create_split_dir.pl /export/b0{1,2,3,4}/$USER/kaldi-data/rm-$(date +'%m_%d_%H_%M')/s5/$dir/wsj_egs/storage $dir/wsj_egs/storage
  fi

  # Create RM features to match the WSJ features
  steps/online/nnet2/get_egs2.sh --cmd "$train_cmd" \
    data/train exp/tri3b_ali ${src_dir} $dir/wsj_egs
fi

if [ $stage -le 2 ]; then
  # Language zero will be WSJ, since that's where we have the initial system, 
  # and language 1 will be RM.

  steps/nnet2/train_multilang2.sh \
    --cleanup false --num-epochs 4 \
    --initial-learning-rate 0.04 --final-learning-rate 0.008 \
    --mix-up "10000 8000" \
    --cmd "$train_cmd" --num-threads 1 --num-jobs-nnet "6 3" --parallel-opts "-l gpu=1 -q g.q" \
    $src_alidir $src_dir/egs \
     exp/tri5b_ali $dir/wsj_egs \
    $src_dir/10.mdl $dir
fi

if [ $stage -le 3 ]; then
  steps/online/nnet2/prepare_online_decoding_transfer.sh \
    ${src_dir} data/lang ${dir}/1 ${dir}_online
fi

if [ $stage -le 4 ]; then
  # do online decoding with the resulting model
  steps/online/nnet2/decode.sh --config conf/decode.config --cmd "$decode_cmd" --nj 20 \
    exp/tri3b/graph data/test ${dir}_online/decode &
  steps/online/nnet2/decode.sh --config conf/decode.config --cmd "$decode_cmd" --nj 20 \
    exp/tri3b/graph_ug data/test ${dir}_online/decode_ug || exit 1;
  wait
fi

#steps/nnet2/train_multilang2.sh: Will train for 7 epochs (of language 0) = 345 iterations
#steps/nnet2/train_multilang2.sh: 345 iterations is approximately 12 epochs for language 1

