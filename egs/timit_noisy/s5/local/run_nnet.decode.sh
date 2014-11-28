. path.sh
. cmd.sh

decode_nj=16
transform_dir=

. parse_options.sh

if [ $# -ne 3 ]; then
  echo "Usage: local/run_nnet.decode.sh <data-dir> <graph-dir> <model-dir>"
  echo "e.g. : local/run_nnet.decode.sh data/test_clean exp/tri4_nnet_clean"
fi

test=$1
graph_dir=$2
model_dir=$3

testid=`basename $test`

transform_dir_opts=
[ ! -z "$transform_dir" ] && transform_dir_opts="--transform-dir $transform_dir"

decode_extra_opts=(--num-threads 6 --parallel-opts "-pe smp 6 -l mem_free=4G,ram_free=0.7G")

steps/nnet2/decode.sh --cmd "$decode_cmd" --nj "$decode_nj" "${decode_extra_opts[@]}" \
  $transform_dir_opts $graph_dir $test \
  $model_dir/decode_${testid} || exit 1 
touch $model_dir/decode_$testid/.done
