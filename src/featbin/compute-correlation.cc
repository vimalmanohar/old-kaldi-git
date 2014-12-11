// featbin/compute-correlation.cc

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

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;

    const char *usage =
        "Computes the correlation between two sets of features "
        "using the CCA objective. \n"
        "The target dim can be set using --target-dim.\n"
        "\n"
        "Usage: compute-correlation <in-rspecifier1> <in-rspecifier2> [<out-trans-12> <out-trans-21>]\n"
        "e.g. : compare-correlation ark:1.ark ark:2.ark\n";

    ParseOptions po(usage);

    int32 target_dim = 0;
    bool binary = true;
    std::string r = "1.0:1.0";  // Colon-separated list of regularizer for different views

    po.Register("target-dim", &target_dim, "Compute the correlation with only "
                "the top target_dim CCA dimensions");
    po.Register("regularizer-list", &r, "Colon-separated list of regularizer "
                 "for different views; Required to ensure "
                 "positive definiteness of covariance matrices");
    po.Register("binary", &binary, "Write in binary mode (only relevant if output is a wxfilename)");
  
    po.Read(argc, argv);

    if (po.NumArgs() != 2 && po.NumArgs() != 4) {
      po.PrintUsage();
      exit(1);
    }
    
    std::vector<BaseFloat> regularizers;
    SplitStringToFloats(r, ":", false, &regularizers);

    KALDI_ASSERT(regularizers.size() == 2);

    std::string rspecifier1 = po.GetArg(1), rspecifier2 = po.GetArg(2);
    
    int32 num_done = 0, num_err = 0, dim = 0;
    int64 num_feats = 0;

    SequentialBaseFloatMatrixReader feat_reader1(rspecifier1);
    RandomAccessBaseFloatMatrixReader feat_reader2(rspecifier2);

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
        if (target_dim == 0) target_dim = dim;
        int32 len = feat1.NumRows();
        Sigma_1.Resize(dim);
        Sigma_2.Resize(dim);
        Sigma_12.Resize(dim,dim);
        Sigma_12.AddMatMat(1.0/(len-1), feat1, kTrans, feat2, kNoTrans, 1.0);
        Sigma_1.AddMat2(1.0/(len-1), feat1, kTrans, 1.0);
        Sigma_2.AddMat2(1.0/(len-1), feat2, kTrans, 1.0);
      }
    

      num_done++;
      num_feats += feat1.NumRows();
    }

    Sigma_12.Scale(1.0/(num_feats-1));
    Sigma_1.Scale(1.0/(num_feats-1));
    Sigma_2.Scale(1.0/(num_feats-1));

    Sigma_1.AddToDiag(regularizers[0]);
    TpMatrix<BaseFloat> R_1(Sigma_1.NumRows());
    R_1.Cholesky(Sigma_1);
    R_1.Invert();
    
    Sigma_2.AddToDiag(regularizers[1]);
    TpMatrix<BaseFloat> R_2(Sigma_2.NumRows());
    R_2.Cholesky(Sigma_2);
    R_2.Invert();

    Matrix<BaseFloat> T(dim,dim);
    T.AddTpMatTp(1.0, R_1, Sigma_12, kNoTrans, 
                  R_2, 0.0);

    int32 min_rc = dim;
    Matrix<BaseFloat> U(dim, min_rc);
    Matrix<BaseFloat> Vt(dim, min_rc);
    Vector<BaseFloat> s(min_rc);
    T.Svd(&s, &U, &Vt);
    SortSvd(&s, &U, &Vt);

    U.Resize(dim, target_dim, kCopyData);
    s.Resize(target_dim, kCopyData);
    Vt.Resize(target_dim, dim, kCopyData);

    BaseFloat corr = s.Sum();

    KALDI_LOG << "The correlation between the two sets of features is " 
              << corr << " using top " << target_dim
              << " CCA dimensions; feature dimension is " << dim;

    if (po.NumArgs() == 4) {
      Matrix<BaseFloat> trans_12(dim, target_dim);
      Matrix<BaseFloat> trans_21(dim, target_dim);

      trans_12.AddSpMat(1.0, Sigma_1, U, kNoTrans, 0.0);
      trans_21.AddSpMat(1.0, Sigma_2, Vt, kTrans, 0.0);

      WriteKaldiObject(trans_12, po.GetArg(3), binary);
      WriteKaldiObject(trans_21, po.GetArg(4), binary);
    }

    KALDI_LOG << "Processed " << num_done << " feature files, "
              << num_err << " had errors.";

    return (num_done > 0 ) ? 0 : 1;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

