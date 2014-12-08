#!/bin/bash 

# Copyright 2012-2014  Johns Hopkins University (Author: Daniel Povey).  Apache 2.0.
# Copyright 2014 Vimal Manohar

# This script does multiview training of neural network with DCCA objective.

# This is a modified version of train_tanh_fast.sh, which also trains a tanh 
# network, but as a DNN rather than a Denoising Autoencoder.
# train_tanh_fast.sh is a new, improved version of train_tanh.sh, which uses
# the 'online' preconditioning method.  For GPUs it's about two times faster
# than before (although that's partly due to optimizations that will also help
# the old recipe), and for CPUs it gives better performance than the old method
# (I believe); also, the difference in optimization performance between CPU and
# GPU is almost gone.  The old train_tanh.sh script is now deprecated.
# We made this a separate script because not all of the options that the
# old script accepted, are still accepted.


# Begin configuration section.
cmd=run.pl
num_epochs=15      # Number of epochs during which we reduce
                   # the learning rate; number of iteration is worked out from this.
num_epochs_extra=5 # Number of epochs after we stop reducing
                   # the learning rate.
num_iters_final=20 # Maximum number of final iterations to give to the
                   # optimization over the validation set.
initial_learning_rate=0.04
final_learning_rate=0.004
bias_stddev=0.5
shrink_interval=5 # shrink every $shrink_interval iters except while we are 
                  # still adding layers, when we do it every iter.
shrink=true
num_frames_shrink=2000 # note: must be <= --num-frames-diagnostic option to get_denoisining_autoencoder_egs.sh, if
                       # given.
final_learning_rate_factor=0.5 # Train the two last layers of parameters half as
                               # fast as the other layers, by default.

hidden_layer_dim=300 #  You may want this larger, e.g. 1024 or 2048.

minibatch_size=128 # by default use a smallish minibatch size for neural net
                   # training; this controls instability which would otherwise
                   # be a problem with multi-threaded update. 

samples_per_iter=200000 # each iteration of training, see this many samples
                        # per job.  This option is passed to get_denoising_autoencoder_egs.sh.
num_jobs_nnet=8    # Number of neural net jobs to run in parallel.  This option
                   # is passed to get_denoising_autoencoder_egs.sh.
get_egs_stage=0

shuffle_buffer_size=5000 # This "buffer_size" variable controls randomization of
                # the samples on each iter.  You could set it to 0 or to a large
                # value for complete randomization, but this would both consume
                # memory and cause spikes in disk I/O.  Smaller is easier on
                # disk and memory but less random.  It's not a huge deal though,
                # as samples are anyway randomized right at the start.

stage=-5

io_opts="-tc 5" # for jobs with a lot of I/O, limits the number running at one time.   These don't
splice_width=3 # meaning +- 3 frames on each side for second LDA
randprune=4.0 # speeds up LDA.
alpha=4.0 # relates to preconditioning.
update_period=4 # relates to online preconditioning: says how often we update the subspace.
num_samples_history=2000 # relates to online preconditioning
max_change_per_sample=0.075
# we make the [input, output] ranks less different for the tanh setup than for
# the pnorm setup, as we don't have the difference in dimensions to deal with.
precondition_rank_in=30  # relates to online preconditioning
precondition_rank_out=60 # relates to online preconditioning
num_threads=16
parallel_opts="-pe smp 16 -l ram_free=1G,mem_free=1G" # by default we use 16 threads; this lets the queue know.
  # note: parallel_opts doesn't automatically get adjusted if you adjust num-threads.
combine_parallel_opts="-pe smp 8"  # queue options for the "combine" stage.
combine_num_threads=8
cleanup=false
egs_dir=
egs_opts=
transform_dir_view1=
transform_dir_view2=
cmvn_opts=  # will be passed to get_denoising_autoencoder_egs.sh, if supplied.  
            # only relevant for "raw" features, not lda.
nj=4
feat_type=raw  # Can be used to force "raw" features.
prior_subset_size=10000 # 10k samples per job, for computing priors.  Should be
                        # more than enough.
regularizer_list="1.0:1.0"
output_dim_list="13 13"

# End configuration section.

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;


if [ $# != 5 ]; then
  echo "Usage: $0 [opts] <in-data> <out-data> <exp-dir>"
  echo " e.g.: $0 data/train_multi data/train_clean exp/nnet_da"
  echo ""
  echo "Main options (for others, see top of script file)"
  echo "  --config <config-file>                           # config file containing options"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --num-epochs <#epochs|15>                        # Number of epochs of main training"
  echo "                                                   # while reducing learning rate (determines #iterations, together"
  echo "                                                   # with --samples-per-iter and --num-jobs-nnet)"
  echo "  --num-epochs-extra <#epochs-extra|5>             # Number of extra epochs of training"
  echo "                                                   # after learning rate fully reduced"
  echo "  --initial-learning-rate <initial-learning-rate|0.02> # Learning rate at start of training, e.g. 0.02 for small"
  echo "                                                       # data, 0.01 for large data"
  echo "  --final-learning-rate  <final-learning-rate|0.004>   # Learning rate at end of training, e.g. 0.004 for small"
  echo "                                                   # data, 0.001 for large data"
  echo "  --num-hidden-layers <#hidden-layers|2>           # Number of hidden layers, e.g. 2 for 3 hours of data, 4 for 100hrs"
  echo "  --initial-num-hidden-layers <#hidden-layers|1>   # Number of hidden layers to start with."
  echo "  --add-layers-period <#iters|2>                   # Number of iterations between adding hidden layers"
  echo "  --num-jobs-nnet <num-jobs|8>                     # Number of parallel jobs to use for main neural net"
  echo "                                                   # training (will affect results as well as speed; try 8, 16)"
  echo "                                                   # Note: if you increase this, you may want to also increase"
  echo "                                                   # the learning rate."
  echo "  --num-threads <num-threads|16>                   # Number of parallel threads per job (will affect results"
  echo "                                                   # as well as speed; may interact with batch size; if you increase"
  echo "                                                   # this, you may want to decrease the batch size."
  echo "  --parallel-opts <opts|\"-pe smp 16 -l ram_free=1G,mem_free=1G\">      # extra options to pass to e.g. queue.pl for processes that"
  echo "                                                   # use multiple threads... note, you might have to reduce mem_free,ram_free"
  echo "                                                   # versus your defaults, because it gets multiplied by the -pe smp argument."
  echo "  --io-opts <opts|\"-tc 10\">                      # Options given to e.g. queue.pl for jobs that do a lot of I/O."
  echo "  --minibatch-size <minibatch-size|128>            # Size of minibatch to process (note: product with --num-threads"
  echo "                                                   # should not get too large, e.g. >2k)."
  echo "  --samples-per-iter <#samples|200000>             # Number of samples of data to process per iteration, per"
  echo "                                                   # process."
  echo "  --splice-width <width|4>                         # Number of frames on each side to append for feature input"
  echo "                                                   # (note: we splice processed, typically 40-dimensional frames"
  echo "  --num-iters-final <#iters|20>                    # Number of final iterations to give to nnet-combine-fast to "
  echo "                                                   # interpolate parameters (the weights are learned with a validation set)"
  echo "  --stage <stage|-9>                               # Used to run a partially-completed training process from somewhere in"
  echo "                                                   # the middle."
  
  exit 1;
fi

data_view1=$1
data_view2=$2
model_view1=$3
model_view2=$4
dir=$5

# Check some files.
for f in $data_view1/feats.scp $data_view2/feats.scp; do
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done

mkdir -p $dir/log

extra_opts=()
[ ! -z "$cmvn_opts" ] && extra_opts+=(--cmvn-opts "$cmvn_opts")
[ ! -z "$feat_type" ] && extra_opts+=(--feat-type $feat_type)
[ ! -z "$online_ivector_dir" ] && extra_opts+=(--online-ivector-dir $online_ivector_dir)
extra_opts+=(--transform-dir-view1 "$transform_dir_view1")
extra_opts+=(--transform-dir-view2 "$transform_dir_view2")
extra_opts+=(--splice-width $splice_width)

if [ $stage -le -3 ] && [ -z "$egs_dir" ]; then
  echo "$0: calling get_multiview_egs.sh"
  local/nnet2/get_multiview_egs.sh $egs_opts \
    "${extra_opts[@]}" --nj $nj \
    --samples-per-iter $samples_per_iter \
    --num-jobs-nnet $num_jobs_nnet --stage $get_egs_stage \
    --cmd "$cmd" $egs_opts --io-opts "$io_opts" \
    $data_view1 $data_view2 $dir || exit 1;
fi
if [ -z $egs_dir ]; then
  egs_dir=$dir/egs
fi

iters_per_epoch=`cat $egs_dir/iters_per_epoch`  || exit 1;
! [ $num_jobs_nnet -eq `cat $egs_dir/num_jobs_nnet` ] && \
  echo "$0: Warning: using --num-jobs-nnet=`cat $egs_dir/num_jobs_nnet` from $egs_dir"
num_jobs_nnet=`cat $egs_dir/num_jobs_nnet` || exit 1;

if [ $stage -le -2 ]; then
  echo "$0: initializing neural net";
  
  online_preconditioning_opts="alpha=$alpha num-samples-history=$num_samples_history update-period=$update_period rank-in=$precondition_rank_in rank-out=$precondition_rank_out max-change-per-sample=$max_change_per_sample"
  
  nc=`nnet2-info --raw=true $model_view1 2> /dev/null | grep num-components | awk '{print $2}'` || exit 1
  [ -z "$nc" ] && echo "$0: Unable to parse num-components from nnet2-info --raw=true $model_view1" && exit 1

  hidden_layer_dim=`nnet2-info --raw=true $model_view1 2> /dev/null | grep "component $[nc-1]" | perl -pe 's/.+input-dim=(\d+), .+/$1/'` || exit 1
  [ -z "$hidden_layer_dim" ] && echo "$0: Unable to parse hidden_layer_dim from nnet2-info --raw=true $model_view1" && exit 1

  stddev=`perl -e "print 1.0/sqrt($hidden_layer_dim);"`

  output_dim=`echo $output_dim_list | awk '{print $1}'`
  cat >$dir/nnet.view1.config <<EOF
AffineComponentPreconditionedOnline input-dim=$hidden_layer_dim output-dim=$output_dim $online_preconditioning_opts learning-rate=$initial_learning_rate param-stddev=$stddev bias-stddev=$bias_stddev
EOF

  $cmd $dir/log/nnet_init.view1.log \
    nnet2-copy --raw=true --truncate=$[nc-1] $model_view1 - \| \
    nnet-insert --raw=true --insert-at=$[nc-1] --randomize-next-component=false \
    - "nnet-init $dir/nnet.view1.config - |" $dir/0.view1.nnet || exit 1;

  nc=`nnet2-info --raw=true $model_view2 2> /dev/null | grep num-components | awk '{print $2}'` || exit 1
  [ -z "$nc" ] && echo "$0: Unable to parse num-components from nnet2-info --raw=true $model_view2" && exit 1

  hidden_layer_dim=`nnet2-info --raw=true $model_view2 2> /dev/null | grep "component $[nc-1]" | perl -pe 's/.+input-dim=(\d+), .+/$1/'` || exit 1
  [ -z "$hidden_layer_dim" ] && echo "$0: Unable to parse hidden_layer_dim from nnet2-info --raw=true $model_view2" && exit 1
  stddev=`perl -e "print 1.0/sqrt($hidden_layer_dim);"`

  output_dim=`echo $output_dim_list | awk '{print $2}'`
  cat >$dir/nnet.view2.config <<EOF
AffineComponentPreconditionedOnline input-dim=$hidden_layer_dim output-dim=$output_dim $online_preconditioning_opts learning-rate=$initial_learning_rate param-stddev=$stddev bias-stddev=$bias_stddev
EOF

  $cmd $dir/log/nnet_init.view2.log \
    nnet2-copy --raw=true --truncate=$[nc-1] $model_view2 - \| \
    nnet-insert --raw=true --insert-at=$[nc-1] --randomize-next-component=false \
    - "nnet-init $dir/nnet.view2.config - |" $dir/0.view2.nnet || exit 1;
fi

num_iters_reduce=$[$num_epochs * $iters_per_epoch];
num_iters_extra=$[$num_epochs_extra * $iters_per_epoch];
num_iters=$[$num_iters_reduce+$num_iters_extra]

echo "$0: Will train for $num_epochs + $num_epochs_extra epochs, equalling "
echo "$0: $num_iters_reduce + $num_iters_extra = $num_iters iterations, "
echo "$0: (while reducing learning rate) + (with constant learning rate)."

x=0
while [ $x -lt $num_iters ]; do
  if [ $x -ge 0 ] && [ $stage -le $x ]; then

    # Set off jobs doing some diagnostics, in the background.
    $cmd $dir/log/compute_corr_valid.$x.log \
      nnet-compute-corr --raw=true $dir/$x.view1.nnet $dir/$x.view2.nnet ark:$egs_dir/valid_diagnostic.egs &
    $cmd $dir/log/compute_corr_train.$x.log \
      nnet-compute-corr --raw=true $dir/$x.view1.nnet $dir/$x.view2.nnet ark:$egs_dir/train_diagnostic.egs &

    #if [ $x -gt 0 ]; then
    #  $cmd $dir/log/progress.$x.log \
    #    nnet-show-progress --raw=true --use-gpu=no $dir/$[$x-1].nnet $dir/$x.nnet \
    #     ark:$egs_dir/train_diagnostic.egs '&&' \
    #     nnet2-info --raw=true $dir/$x.nnet &
    #fi

    echo "Training neural net (pass $x)"
    
    this_minibatch_size=$minibatch_size
    do_average=true

    $cmd $parallel_opts JOB=1:$num_jobs_nnet $dir/log/train.$x.JOB.log \
        nnet-shuffle-multiview-egs --buffer-size=$shuffle_buffer_size --srand=$x \
        ark:$egs_dir/egs.JOB.$[$x%$iters_per_epoch].ark ark:- \| \
        nnet-train-dcca --minibatch-size=$this_minibatch_size --srand=$x \
        --raw=true --regularizer-list="$regularizer_list" \
        $dir/$x.view1.nnet $dir/$x.view2.nnet ark:- \
        $dir/$[$x+1].JOB.view1.nnet $dir/$[$x+1].JOB.view2.nnet || exit 1;

    nnets_list_view1=
    for n in `seq 1 $num_jobs_nnet`; do
      nnets_list_view1="$nnets_list_view1 $dir/$[$x+1].$n.view1.nnet"
    done
    
    nnets_list_view2=
    for n in `seq 1 $num_jobs_nnet`; do
      nnets_list_view2="$nnets_list_view2 $dir/$[$x+1].$n.view2.nnet"
    done

    learning_rate=`perl -e '($x,$n,$i,$f)=@ARGV; print ($x >= $n ? $f : $i*exp($x*log($f/$i)/$n));' $[$x+1] $num_iters_reduce $initial_learning_rate $final_learning_rate`;
    last_layer_learning_rate=`perl -e "print $learning_rate * $final_learning_rate_factor;"`;
    nnet2-info --raw=true $dir/$[$x+1].1.view1.nnet> $dir/foo  2>/dev/null || exit 1
    nu=`cat $dir/foo | grep num-updatable-components | awk '{print $2}'`
    na=`cat $dir/foo | grep -v Fixed | grep AffineComponent | wc -l` 
    # na is number of last updatable AffineComponent layer [one-based, counting only
    # updatable components.]
    # The last two layers will get this (usually lower) learning rate.
    lr_string="$learning_rate"
    for n in `seq 2 $nu`; do 
      if [ $n -eq $na ] || [ $n -eq $[$na-1] ]; then lr=$last_layer_learning_rate;
      else lr=$learning_rate; fi
      lr_string="$lr_string:$lr"
    done

    $cmd $dir/log/average_view1.$x.log \
      nnet-average --raw=true $nnets_list_view1 - \| \
      nnet2-copy --raw=true --learning-rates=$lr_string - $dir/$[$x+1].view1.nnet || exit 1;
    
    learning_rate=`perl -e '($x,$n,$i,$f)=@ARGV; print ($x >= $n ? $f : $i*exp($x*log($f/$i)/$n));' $[$x+1] $num_iters_reduce $initial_learning_rate $final_learning_rate`;
    last_layer_learning_rate=`perl -e "print $learning_rate * $final_learning_rate_factor;"`;
    nnet2-info --raw=true $dir/$[$x+1].1.view2.nnet> $dir/foo  2>/dev/null || exit 1
    nu=`cat $dir/foo | grep num-updatable-components | awk '{print $2}'`
    na=`cat $dir/foo | grep -v Fixed | grep AffineComponent | wc -l` 
    # na is number of last updatable AffineComponent layer [one-based, counting only
    # updatable components.]
    # The last two layers will get this (usually lower) learning rate.
    lr_string="$learning_rate"
    for n in `seq 2 $nu`; do 
      if [ $n -eq $na ] || [ $n -eq $[$na-1] ]; then lr=$last_layer_learning_rate;
      else lr=$learning_rate; fi
      lr_string="$lr_string:$lr"
    done
    
    $cmd $dir/log/average_view2.$x.log \
      nnet-average --raw=true $nnets_list_view2 - \| \
      nnet2-copy --raw=true --learning-rates=$lr_string - $dir/$[$x+1].view2.nnet || exit 1;
    
    <<COMMENT
    if $shrink && [ $[$x % $shrink_interval] -eq 0 ]; then
      mb=$[($num_frames_shrink+$num_threads-1)/$num_threads]
      $cmd $combine_parallel_opts $dir/log/shrink.$x.log \
        nnet-subset-egs --n=$num_frames_shrink --randomize-order=true --srand=$x \
          ark:$egs_dir/train_diagnostic.egs ark:-  \| \
        nnet-combine-fast --raw=true --use-gpu=no --num-threads=$combine_num_threads \
          --verbose=3 --minibatch-size=$mb $objf_opts \
          $dir/$[$x+1].nnet ark:- $dir/$[$x+1].nnet || exit 1;
    else
      # On other iters, do nnet-am-fix which is much faster and has roughly
      # the same effect.
      nnet-fix --raw=true $dir/$[$x+1].nnet $dir/$[$x+1].nnet 2>$dir/log/fix.$x.log 
    fi
COMMENT

    rm $nnets_list_view1 $nnets_list_view2
  fi
  x=$[$x+1]
done

ln -sf $x.view1.nnet $dir/final.view1.nnet
ln -sf $x.view2.nnet $dir/final.view2.nnet

<<COMMENT
# Now do combination.
# At the end, final.nnet will be a combination of the last e.g. 10 models.
nnets_list_view1=()
if [ $num_iters_final -gt $num_iters_extra ]; then
  echo "Setting num_iters_final=$num_iters_extra"
fi
start=$[$num_iters-$num_iters_final+1]
for x in `seq $start $num_iters`; do
  idx=$[$x-$start]
  nnets_list_view1[$idx]=$dir/$x.view1.nnet # "nnet2-copy --raw=true --remove-dropout=true $dir/$x.nnet - |"
  nnets_list_view1[$idx]=$dir/$x.view2.nnet # "nnet2-copy --raw=true --remove-dropout=true $dir/$x.nnet - |"
done

if [ $stage -le $num_iters ]; then
  echo "Doing final combination to produce final.nnet"
  # Below, use --use-gpu=no to disable nnet-combine-fast from using a GPU, as if
  # there are many models it can give out-of-memory error on the GPU; set
  # num-threads to 8 to speed it up (this isn't ideal...)
  num_egs=`nnet-copy-egs ark:$egs_dir/combine.egs ark:/dev/null 2>&1 | tail -n 1 | awk '{print $NF}'`
  mb=$[($num_egs+$combine_num_threads-1)/$combine_num_threads]
  [ $mb -gt 512 ] && mb=512
  $cmd $combine_parallel_opts $dir/log/combine.log \
    nnet-combine-fast --raw=true --use-gpu=no --num-threads=$combine_num_threads \
      --verbose=3 --minibatch-size=$mb $objf_opts \
      "${nnets_list_view1[@]}" ark:$egs_dir/combine.egs \
      $dir/final.nnet || exit 1;

  # Compute the probability of the final, combined model with
  # the same subset we used for the previous compute_probs, as the
  # different subsets will lead to different probs.
  $cmd $dir/log/compute_prob_valid.final.log \
    nnet-compute-prob --raw=true $objf_opts $dir/final.nnet ark:$egs_dir/valid_diagnostic.egs &
  $cmd $dir/log/compute_prob_train.final.log \
    nnet-compute-prob --raw=true $objf_opts $dir/final.nnet ark:$egs_dir/train_diagnostic.egs &
fi
COMMENT

sleep 2

echo Done

if $cleanup; then
  echo Cleaning up data
  if [ $egs_dir == "$dir/egs" ]; then
    steps/nnet2/remove_egs.sh $dir/egs
  fi
  echo Removing most of the models
  for x in `seq 0 $num_iters`; do
    if [ $[$x%100] -ne 0 ] && [ $x -lt $[$num_iters-$num_iters_final+1] ]; then 
       # delete all but every 10th model; don't delete the ones which combine to form the final model.
      rm $dir/$x.nnet
    fi
  done
fi

