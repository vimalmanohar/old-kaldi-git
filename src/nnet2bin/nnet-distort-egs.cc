// nnet2bin/nnet2-distort-egs.cc

// Copyright 2014  Pegah Ghahremani
// Copyright 2014  Vimal Manohar

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "hmm/transition-model.h"
#include "transform/cmvn.h"
#include "nnet2/nnet-example-functions.h"

namespace kaldi {
  
  void GetCovarianceFromCmvnStats(const Matrix<BaseFloat> &stats,
                                  Vector<BaseFloat> *cov) {
    int32 dim = stats.NumCols() - 1;
    BaseFloat count = stats(0, dim);
    
    SubVector<BaseFloat> row_0(stats,0);
    SubVector<BaseFloat> x_stats(row_0, 0, dim);

    if (stats.NumRows() == 1) {
      KALDI_ERR << "Variance stats not found in CMVN stats";
    }

    SubVector<BaseFloat> row_1(stats,1);
    SubVector<BaseFloat> x2_stats(row_1, 0, dim);

    cov->Resize(dim);
    cov->AddVec(1.0/count, x2_stats);
    cov->AddVec2(-1.0/count/count, x_stats);
  }
}

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet2;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    const char *usage =
        "This program generates distorted examples \n"
        "by perturbing the features by adding Gaussian noise to it\n"
        "Usage: nnet-distort-egs [options] <egs-rspecifier> <egs-wspecifier> \n"
        "e.g.:\n"
        "nnet-distort-egs ark:train.egs ark:distorted.egs\n";
    
    BaseFloat variance = 1.0;
    std::string cmvn_stats_rxfilename;

    ParseOptions po(usage);
    
    po.Register("variance", &variance, "Variance of the Gaussian noise to be "
                "added to the egs");
    po.Register("cmvn-stats", &cmvn_stats_rxfilename, "CMVN Global stats; "
                "If supplied, add noise of this covariance");

    po.Read(argc, argv);    
    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }
    
    std::string examples_rspecifier = po.GetArg(1),
                examples_wspecifier = po.GetArg(2);
    SequentialNnetExampleReader example_reader(examples_rspecifier); 
    NnetExampleWriter example_writer(examples_wspecifier);
    
    Vector<BaseFloat> covariance;
    if (cmvn_stats_rxfilename != "") {
      bool binary;
      Input ki(cmvn_stats_rxfilename, &binary);
      Matrix<BaseFloat> cmvn_stats;
      cmvn_stats.Read(ki.Stream(), binary);
      GetCovarianceFromCmvnStats(cmvn_stats, &covariance);
    }

    int32 num_done = 0, num_err = 0; 
    for (; !example_reader.Done(); example_reader.Next(), num_done++) {
      NnetExample eg = example_reader.Value();
      Matrix<BaseFloat> input_frames(eg.input_frames);
      Matrix<BaseFloat> distortion(input_frames.NumRows(), input_frames.NumCols());
      distortion.SetRandn();
      if (cmvn_stats_rxfilename != "") {
        if (covariance.Dim() != input_frames.NumCols()) {
          KALDI_ERR << "Dimension mismatch between egs and covariance; "
                    << input_frames.NumCols() << " vs " << covariance.Dim();
        }
        distortion.MulColsVec(covariance);
      }
      input_frames.AddMat(variance, distortion);
      eg.input_frames.CopyFromMat(input_frames);
      example_writer.Write(example_reader.Key(), eg);
    }
    KALDI_LOG << "Successfully processed " << num_done
              << " examples, " << num_err << " with errors.";
    return (num_done == 0 ? 1 : 0);
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}
