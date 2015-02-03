// featbin/acc-cca-stats.cc

// Copyright 2014   Vimal Manohar

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
#include "matrix/kaldi-matrix.h"
#include "matrix/kaldi-vector.h"
#include "feat/cca.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;

    const char *usage =
        "Accumulates statistics for CCA\n"
        "The target dim can be set using --target-dim.\n"
        "\n"
        "Usage: acc-cca-stats <in-rspecifier1> <in-rspecifier2> <cca-stats>\n"
        "e.g. : acc-cca-stats ark:1.ark ark:2.ark stats\n";

    ParseOptions po(usage);

    bool binary = true;

    po.Register("binary", &binary, "Write in binary mode (only relevant if output is a wxfilename)");
  
    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }
    
    std::string rspecifier1 = po.GetArg(1), rspecifier2 = po.GetArg(2);
                
    int32 num_done = 0, num_err = 0, dim = 0;
    int64 num_feats = 0;

    SequentialBaseFloatMatrixReader feat_reader1(rspecifier1);
    RandomAccessBaseFloatMatrixReader feat_reader2(rspecifier2);

    CcaStats stats;

    Vector<BaseFloat> mu_1;
    Vector<BaseFloat> mu_2;
    SpMatrix<BaseFloat> Sigma_1;
    SpMatrix<BaseFloat> Sigma_2;
    Matrix<BaseFloat> Sigma_12;
    
    for (; !feat_reader1.Done(); feat_reader1.Next()) {
      std::string utt = feat_reader1.Key();
      Matrix<BaseFloat> feat1 (feat_reader1.Value());

      if (!feat_reader2.HasKey(utt)) {
        KALDI_WARN << "Second table has no feature for utterance "
                   << utt;
        num_err++;
        continue;
      }
      Matrix<BaseFloat> feat2 (feat_reader2.Value(utt));
      if (feat1.NumCols() != feat2.NumCols()) {
        KALDI_WARN << "Feature dimensions differ for utterance "
                   << utt << ", " << feat1.NumCols() << " vs. "
                   << feat2.NumCols() << ", skipping  utterance."
                   << utt;
        num_err++;
        continue;
      }
      
      if (feat1.NumRows() != feat2.NumRows()) {
        KALDI_WARN << "Length of feats differ for utterance "
                   << utt << ", " << feat1.NumRows() << " vs. "
                   << feat2.NumRows() << ", skipping  utterance."
                   << utt;
        num_err++;
        continue;
      }
      
      if (num_done == 0){
        dim = feat1.NumCols();
        stats.Resize(dim);
      }
      
      stats.Accumulate(feat1,feat2);
      
      num_done++;
      num_feats += feat1.NumRows();
    }

    Output ko(po.GetArg(3), binary);
    stats.Write(ko.Stream(), binary);

    KALDI_LOG << "Processed " << num_done << " feature files, "
              << num_err << " had errors.";

    return (num_done > 0 ) ? 0 : 1;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

