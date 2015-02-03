set -o pipefail

suffix=

. path.sh
. utils/parse_options.sh

if [ $# -ne 1 ]; then 
  echo "Incorrect number of options: $#; Expecting 1"
  exit 1
fi

nnet_dir=$1

mkdir -p data_dcca$suffix data_append$suffix

x=train_multi
if [ ! -f data_dcca$suffix/$x/.dcca.done ]; then
  steps/nnet2/dump_posteriors.sh --iter final.view2 --cmd queue.pl --nj 100 \
    --raw true data_noisy/$x $nnet_dir \
    exp/make_dcca_feats$suffix dcca_feats$suffix data_dcca$suffix/$x

  steps/compute_cmvn_stats.sh data_dcca$suffix/$x exp/make_dcca_feats$suffix/$x dcca_feats$suffix

  touch data_dcca$suffix/$x/.dcca.done
fi

if [ ! -f data_append$suffix/$x/.append.done ]; then
  steps/append_feats.sh --cmd queue.pl --nj 100 \
    data_noisy/$x data_dcca$suffix/$x data_append$suffix/$x \
    exp/append_mfcc_dcca$suffix append_feats$suffix

  steps/compute_cmvn_stats.sh data_append$suffix/$x exp/append_mfcc_dcca$suffix/$x append_feats$suffix

  touch data_append$suffix/$x/.append.done
fi

x=train_clean_multi
if [ ! -f data_dcca$suffix/$x/.dcca.done ]; then
  steps/nnet2/dump_posteriors.sh --iter final.view1 --cmd queue.pl --nj 100 \
    --raw true data_clean/$x $nnet_dir \
    exp/make_dcca_feats$suffix dcca_feats$suffix data_dcca$suffix/$x

  steps/compute_cmvn_stats.sh data_dcca$suffix/$x exp/make_dcca_feats$suffix/$x dcca_feats$suffix

  touch data_dcca$suffix/$x/.dcca.done
fi

for t in dev test; do
  while read noise_type <&3; do
    while read snr <&4; do
      x=${t}_noisy_${noise_type}_snr_$snr

      if [ ! -f data_dcca$suffix/$x/.dcca.done ]; then
        steps/nnet2/dump_posteriors.sh --iter final.view2 --cmd queue.pl --nj 10 \
          --raw true data_noisy/$x $nnet_dir \
          exp/make_dcca_feats$suffix dcca_feats$suffix data_dcca$suffix/$x

        steps/compute_cmvn_stats.sh data_dcca$suffix/$x exp/make_dcca_feats$suffix/$x dcca_feats$suffix

        touch data_dcca$suffix/$x/.dcca.done
      fi

      if [ ! -f data_append$suffix/$x/.append.done ]; then
        steps/append_feats.sh --cmd queue.pl --nj 10 \
          data_noisy/$x data_dcca$suffix/$x data_append$suffix/$x \
          exp/append_mfcc_dcca$suffix append_feats$suffix

        steps/compute_cmvn_stats.sh data_append$suffix/$x exp/append_mfcc_dcca$suffix/$x append_feats$suffix

        touch data_append$suffix/$x/.append.done
      fi

      x=${t}_clean_noisy_${noise_type}_snr_$snr

      if [ ! -f data_dcca$suffix/$x/.dcca.done ]; then
        steps/nnet2/dump_posteriors.sh --iter final.view1 --cmd queue.pl --nj 10 \
          --raw true data_clean/$x $nnet_dir \
          exp/make_dcca_feats$suffix dcca_feats$suffix data_dcca$suffix/$x

        steps/compute_cmvn_stats.sh data_dcca$suffix/$x exp/make_dcca_feats$suffix/$x dcca_feats$suffix

        touch data_dcca$suffix/$x/.dcca.done
      fi

    done 4< conf/snr.list
  done 3< conf/noisetypes.list
done 
