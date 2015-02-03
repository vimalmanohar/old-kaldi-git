// featbin/est-cca.cc

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
        "Computes the correlation from stats "
        "using the CCA objective. \n"
        "The target dim can be set using --target-dim.\n"
        "\n"
        "Usage: est-cca <stats> <out-trans-12> <out-trans-21> [<cmvn_stats_view1> <cmvn_stats_view2>]\n"
        "e.g. : est-cca stats ark:1.ark ark:2.ark\n";

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

    if (po.NumArgs() != 3 && po.NumArgs() != 5) {
      po.PrintUsage();
      exit(1);
    }
    
    std::vector<BaseFloat> regularizers;
    SplitStringToFloats(r, ":", false, &regularizers);

    KALDI_ASSERT(regularizers.size() == 2);

    CcaStats stats;

    {
      bool binary;
      Input ki(po.GetArg(1), &binary);
      stats.Read(ki.Stream(), binary);
    }

    int32 m = stats.NumFeats(), dim = stats.Dim();

    Vector<BaseFloat> mu_1(stats.x1());
    mu_1.Scale(1.0/m);
    Vector<BaseFloat> mu_2(stats.y1());
    mu_2.Scale(1.0/m);

    Matrix<BaseFloat> Sigma_12(stats.xy());
    Sigma_12.AddVecVec(-1.0, mu_1, mu_2);
    Sigma_12.Scale(1.0/(m-1));

    SpMatrix<BaseFloat> Sigma_1(stats.x2());
    Sigma_1.AddVec2(-1.0, mu_1);
    Sigma_1.Scale(1.0/(m-1));
    SpMatrix<BaseFloat> Sigma_2(stats.y2());
    Sigma_2.AddVec2(-1.0, mu_2);
    Sigma_2.Scale(1.0/(m-1));
    
    Sigma_1.AddToDiag(regularizers[0]);
    Sigma_2.AddToDiag(regularizers[1]);

    TpMatrix<BaseFloat> R_1(Sigma_1.NumRows());
    R_1.Cholesky(Sigma_1);
    R_1.Invert();
    
    KALDI_VLOG(4) << "Sigma_1: " << Sigma_1;
    KALDI_VLOG(4) << "Sigma_2: " << Sigma_2;
    KALDI_VLOG(4) << "Sigma_12: " << Sigma_12;
    
    KALDI_VLOG(4) << "R_1_inv: " << R_1;

    Sigma_2.Invert();
    
    KALDI_VLOG(4) << "Sigma_2_inv: " << Sigma_2;

    //TpMatrix<BaseFloat> R_2(Sigma_2.NumRows());
    //R_2.Cholesky(Sigma_2);
    //R_2.Invert();

    int32 min_rc = dim;
    Matrix<BaseFloat> U(dim, min_rc);
    Vector<BaseFloat> s(min_rc);
    {
      SpMatrix<BaseFloat> A(dim);
      Matrix<BaseFloat> temp(dim,dim);
      temp.AddTpMat(1.0, R_1, kNoTrans, Sigma_12, kNoTrans, 0.0);
      A.AddMat2Sp(1.0, temp, kNoTrans, Sigma_2, 0.0);
      
      KALDI_VLOG(4) << "A: " << A;

      A.Eig(&s, &U);
      SortSvd(&s, &U);
    }
    KALDI_VLOG(4) << "U: " << U;
    KALDI_VLOG(4) << "s: " << s;

    if (target_dim == 0) target_dim = dim;

    U.Resize(dim, target_dim, kCopyData);
    s.Resize(target_dim, kCopyData);
    U.AddTpMat(1.0, R_1, kTrans, Matrix<BaseFloat>(U), kNoTrans, 0.0);
    
    KALDI_VLOG(4) << "Rescaled U: " << U;

    Matrix<BaseFloat> Vt(target_dim, dim);
    Vt.AddMatMat(1.0, U, kTrans, Sigma_12, kNoTrans, 0.0);
    KALDI_VLOG(4) << "Vt: " << Vt;
    Vt.AddMatSp(1.0, Matrix<BaseFloat>(Vt), kNoTrans, Sigma_2, 0.0);
    KALDI_VLOG(4) << "Vt: " << Vt;
    for (MatrixIndexT i = 0; i < target_dim; i++) {
      SubVector<BaseFloat> Vt_i(Vt, i);
      if (s(i) >= 1e-10) {
        s(i) = std::sqrt(s(i));
        Vt_i.Scale(1.0/s(i));
      } else {
        s(i) = 0.0;
      }
    }
    KALDI_VLOG(4) << "Rescaled Vt: " << Vt;
    KALDI_VLOG(4) << "Resclaed s: " << s;
    
    BaseFloat corr = s.Sum();


    KALDI_LOG << "The correlation between the two sets of features is " 
              << corr << " using top " << target_dim
              << " CCA dimensions; feature dimension is " << dim;

    WriteKaldiObject(Matrix<BaseFloat>(U,kTrans), po.GetArg(2), binary);
    WriteKaldiObject(Vt, po.GetArg(3), binary);

    if (po.NumArgs() == 5) {
      Matrix<BaseFloat> cmvn_stats_view1(2,dim+1);
      Matrix<BaseFloat> cmvn_stats_view2(2,dim+1);

      for (int32 i = 0; i < dim; i++) {
        cmvn_stats_view1(0,i) = stats.x1()(i);
        cmvn_stats_view2(0,i) = stats.y1()(i);
        cmvn_stats_view1(1,i) = stats.x2()(i,i);
        cmvn_stats_view2(1,i) = stats.y2()(i,i);
      }
      cmvn_stats_view1(0,dim) = stats.NumFeats();
      cmvn_stats_view2(0,dim) = stats.NumFeats();

      {
        Output ko(po.GetArg(4), binary);
        cmvn_stats_view1.Write(ko.Stream(), binary);
      }
      {
        Output ko(po.GetArg(5), binary);
        cmvn_stats_view2.Write(ko.Stream(), binary);
      }
    }

    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


