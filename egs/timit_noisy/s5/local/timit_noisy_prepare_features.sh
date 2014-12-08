echo ============================================================================
echo "         MFCC Feature Extration & CMVN for Training and Test set           "
echo ============================================================================

type=mfcc
nj=10
exp=exp

. parse_options.sh

mfccdir=mfcc
data=data
done_file=.mfcc.done
mfcc_opts=

if [ $type == hires ]; then
  mfccdir=hires
  data=data_hires
  done_file=.hires.done
  mfcc_opts="--mfcc-config conf/mfcc_hires.conf"
fi

# Make MFCC features for clean data
for x in train dev test; do 
  x=${x}_clean
  if [ ! -f $data/$x/$done_file ]; then
    steps/make_mfcc.sh $mfcc_opts --cmd "$train_cmd" --nj $nj \
      $data/$x $exp/make_$type/$x $mfccdir
    steps/compute_cmvn_stats.sh $data/$x $exp/make_$type/$x $mfccdir || exit 1
    utils/fix_data_dir.sh $data/$x
    touch $data/$x/$done_file
  fi
done

# Make MFCC features for noisy data
while read noise_type <&3; do
  while read snr <&4; do
    for x in train dev test; do 
      x=${x}_noisy_${noise_type}_snr_${snr}
      if [ ! -f $data/$x/$done_file ]; then
        steps/make_mfcc.sh $mfcc_opts --cmd "$train_cmd" --nj $nj \
          $data/$x $exp/make_$type/$x $mfccdir
        steps/compute_cmvn_stats.sh $data/$x $exp/make_$type/$x $mfccdir
        utils/fix_data_dir.sh $data/$x
        touch $data/$x/$done_file
      fi
    done
  done 4< conf/snr.list
done 3< conf/noisetypes.list

