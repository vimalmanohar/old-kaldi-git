#!/bin/bash

# Copyright 2014  Vimal Manohar

. path.sh
. cmd.sh

set -e
set -u
set -o pipefail

datadir=data_noisy_fbank/train_noisy
dir=exp/irm_nnet

irm_scp=data_noisy_fbank/train_noisy/irm_targets.scp
pnorm_input_dim=3000
pnorm_output_dim=300
num_hidden_layers=5
dnn_init_learning_rate=0.0015
dnn_final_learning_rate=0.001
dnn_init_learning_rate2=0.001
dnn_final_learning_rate2=0.0005
dnn_gpu_parallel_opts=(--minibatch-size 512 --max-change 40 --num-jobs-nnet 4 --num-threads 1 \
  --parallel-opts "-l gpu=1" --cmd "queue.pl -q \"all.q@[gb]0[^5]*\" -l arch=*64 -l mem_free=4G,ram_free=2G")
stage=-100
use_subset=true
percentage=20

. parse_options.sh

if $use_subset; then
  datadir_orig=$datadir
  datadir=${datadir_orig}.${percentage}p
  numutts_keep=`perl -e 'print int($ARGV[0]*$ARGV[1]/100)' "$(wc -l < $datadir_orig/feats.scp)" $percentage`
  subset_data_dir.sh $datadir_orig $numutts_keep $datadir
  filter_scp.pl $datadir/feats.scp $datadir_orig/irm_targets.scp > $datadir/irm_targets.scp
  irm_scp=$datadir/irm_targets.scp
fi

if [ ! -f $dir/.done ]; then
  steps/nnet2/train_irm_nnet.sh \
    "${dnn_gpu_parallel_opts[@]}" \
    --pnorm-input-dim $pnorm_input_dim \
    --pnorm-output-dim $pnorm_output_dim \
    --num-hidden-layers $num_hidden_layers \
    --initial-learning-rate $dnn_init_learning_rate \
    --final-learning-rate $dnn_final_learning_rate \
    --irm_scp $irm_scp --nj 64 \
    --stage $stage --cleanup false \
    $datadir $dir || exit 1 
  touch $dir/.done
fi

mkdir -p ${dir}_update_linear

if [ ! -f ${dir}_update_linear/.done ]; then

  nc=`nnet2-info --raw=true $dir/final.nnet | grep "^component [0-9]" | wc -l`
  nnet2-copy --raw=true --truncate=$[nc-1] $dir/final.nnet ${dir}_update_linear/foo.net || exit 1
  source_model=${dir}_update_linear/foo.net

  tmp_dir=${dir}_update_linear/make_irm_targets
  nj=64

  mkdir -p $tmp_dir/split$nj

  utils/split_scp.pl $irm_scp `eval echo $tmp_dir/split$nj/${irm_scp##*/}.{$(seq -s ',' $nj)}`
  $train_cmd JOB=1:$nj $tmp_dir/make_irm_linear.JOB.log \
    matrix-apply-sigmoid --inverse=true scp:$tmp_dir/split$nj/${irm_scp##*/}.JOB \
    ark,scp:`pwd`/${tmp_dir}/irm_targets_linear.JOB.ark,`pwd`/${tmp_dir}/irm_targets_linear.JOB.scp || exit 1

  for n in `seq $nj`; do cat ${tmp_dir}/irm_targets_linear.$n.scp; done | sort -k 1,1 > ${dir}_update_linear/irm_targets_linear.scp

  irm_linear_scp=${dir}_update_linear/irm_targets_linear.scp

  steps/nnet2/update_irm_nnet.sh \
    "${dnn_gpu_parallel_opts[@]}" \
    --initial-learning-rate $dnn_init_learning_rate2 \
    --final-learning-rate $dnn_final_learning_rate2 \
    --irm_scp $irm_linear_scp --nj 64 \
    --stage $stage --cleanup false \
    --obj-func SquaredError \
    $datadir $source_model ${dir}_update_linear || exit 1 
  touch ${dir}_update_linear/.done
fi

