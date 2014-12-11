// featbin/compute-correlation-incremental.cc

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

namespace kaldi {

  void RankOneUpdateSvd(const Vector<BaseFloat> &x, 
                        Matrix<BaseFloat> *U,
                        Vector<BaseFloat> *s,
                        int32 *k,
                        int max_dim = -1) {

    int d = x.Dim();

    SubMatrix<BaseFloat> U_k(*U,0,d,0,*k);
    Vector<BaseFloat> x_proj(d);
    x_proj.AddMatVec(1.0, U_k, kTrans, x, 0.0);

    Vector<BaseFloat> x_perp(x);
    x_perp.AddVec(-1.0, x_proj);

    BaseFloat x_perp_norm = x_perp.Norm(2);
    if (x_perp_norm > 1e-10) {
      x_perp.Scale(1.0/x_perp_norm);

      Matrix<BaseFloat> U_tmp(d,*k+1);
      Matrix<BaseFloat> S_tmp(*k+1,*k+1);
    
      SubMatrix<BaseFloat> U_tmp_k(U_tmp,0,d,0,*k);
      U_tmp_k.CopyFromMat(U_k);
      U_tmp.CopyColFromVec(x_perp, *k+1);

      SubMatrix<BaseFloat> S_tmp_k(S_tmp,0,*k,0,*k);
      for (MatrixIndexT i = 0; i < *k; i++) 
        S_tmp(i,i) = (*s)(i);
      S_tmp_k.AddVecVec(1.0, x_proj, x_proj);

      SubVector<BaseFloat> S_tmp_kp1(S_tmp, *k+1);
      SubVector<BaseFloat> S_tmp_kp1_k(S_tmp_kp1, 0, *k);
      S_tmp_kp1_k.CopyFromVec(x_proj);
      S_tmp_kp1_k.Scale(x_perp_norm);
      S_tmp(*k+1,*k+1) = x_perp_norm * x_perp_norm;
  
      SpMatrix<BaseFloat> S_tmp_sp(*k+1);
      S_tmp_sp.CopyFromMat(S_tmp, kTakeUpper);

      Matrix<BaseFloat> U_bar(*k+1,*k+1);
      Vector<BaseFloat> s_bar(*k+1);

      S_tmp_sp.Eig(&s_bar, &U_bar);
      SortSvd(&s_bar, &U_bar);

      if (max_dim > 0 && U_bar.NumCols() > max_dim) {
        s_bar.Resize(max_dim, kCopyData);
        U_bar.Resize(*k+1, max_dim, kCopyData);
        *k = max_dim;
      } else { *k = U_bar.NumCols(); }

      SubMatrix<BaseFloat> U_kp1(*U,0,d,0,*k);
      U_kp1.AddMatMat(1.0, U_tmp, kNoTrans, U_bar, kNoTrans, 0.0);
      
      SubVector<BaseFloat> s_kp1(*s,0,*k);
      s_kp1.CopyFromVec(s_bar);
    } else {
      Matrix<BaseFloat> U_tmp(d,*k);
      Matrix<BaseFloat> S_tmp(*k,*k);
    
      U_tmp.CopyFromMat(U_k);

      for (MatrixIndexT i = 0; i < *k; i++) S_tmp(i,i) = (*s)(i);
      S_tmp.AddVecVec(1.0, x_proj, x_proj);

      SpMatrix<BaseFloat> S_tmp_sp(*k);
      S_tmp_sp.CopyFromMat(S_tmp, kTakeUpper);

      Matrix<BaseFloat> U_bar(*k,*k);
      Vector<BaseFloat> s_bar(*k);

      S_tmp_sp.Eig(&s_bar, &U_bar);
      SortSvd(&s_bar, &U_bar);

      U_k.AddMatMat(1.0, U_tmp, kNoTrans, U_bar, kNoTrans, 0.0);

      SubVector<BaseFloat> s_k(*s,0,*k);
      s_k.CopyFromVec(s_bar);
    }
  }
}

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;

    const char *usage =
        "Computes the correlation-incremental between two sets of features "
        "using the CCA objective. \n"
        "The target dim can be set using --target-dim.\n"
        "\n"
        "Usage: compute-correlation-incremental <in-rspecifier1> <in-rspecifier2> [<out-trans-12> <out-trans-21>]\n"
        "e.g. : compare-correlation-incremental ark:1.ark ark:2.ark\n";

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
    int32 current_dim = 0;

    SequentialBaseFloatMatrixReader feat_reader1(rspecifier1);
    RandomAccessBaseFloatMatrixReader feat_reader2(rspecifier2);

    Matrix<BaseFloat> U_11;
    Matrix<BaseFloat> Vt_11;
    Vector<BaseFloat> s_11;
    
    Matrix<BaseFloat> U_22;
    Matrix<BaseFloat> Vt_22;
    Vector<BaseFloat> s_22;

    Matrix<BaseFloat> U_12;
    Matrix<BaseFloat> Vt_12;
    Vector<BaseFloat> s_12;
    
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
        U_11.Resize(dim, target_dim+1);
        s_11.Resize(target_dim+1);
        U_22.Resize(dim, target_dim+1);
        s_22.Resize(target_dim+1);
        U_12.Resize(dim, target_dim+1);
        Vt_12.Resize(dim, target_dim+1);
        s_12.Resize(target_dim+1);
      }
    
      for (MatrixIndexT i = 0; i < feat1.NumRows(); i++) {
        Vector<BaseFloat> x1(feat1.Row(i));
        Vector<BaseFloat> x2(feat2.Row(i));
        RankOneUpdateSvd(x1, &U_11, &s_11, &current_dim);
        RankOneUpdateSvd(x2, &U_22, &s_22, &current_dim);
      }
      num_done++;
      num_feats += feat1.NumRows();
    }

    SpMatrix<BaseFloat> Sigma_1_inv_sqrt(target_dim);
    SpMatrix<BaseFloat> Sigma_2_inv_sqrt(target_dim);

    for (MatrixIndexT i = 0; i < target_dim; i++) {
      s_11(i) = 1/std::sqrt(s_11(i)/(num_feats-1) + regularizers[0]);
      s_22(i) = 1/std::sqrt(s_22(i)/(num_feats-1) + regularizers[1]);
    }

    Sigma_1_inv_sqrt.AddMat2Vec(1.0, U_11, kTrans, s_11, 0.0);
    Sigma_2_inv_sqrt.AddMat2Vec(1.0, U_22, kTrans, s_22, 0.0);

    Matrix<BaseFloat> Sigma_12(target_dim, target_dim);
    Matrix<BaseFloat> s_Vt_12(target_dim, target_dim);
    s_Vt_12.AddDiagVecMat(1.0, s_12, Vt_12, kNoTrans,0.0);
    Sigma_12.AddMatMat(1/(num_feats-1), U_12, kNoTrans, s_Vt_12, kNoTrans, 0.0);

    Matrix<BaseFloat> T(dim,dim);
    T.AddSpMatSp(1.0, Sigma_1_inv_sqrt, Sigma_12, kNoTrans, 
                  Sigma_2_inv_sqrt, 0.0);

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

      trans_12.AddSpMat(1.0, Sigma_1_inv_sqrt, U, kNoTrans, 0.0);
      trans_21.AddSpMat(1.0, Sigma_2_inv_sqrt, Vt, kTrans, 0.0);

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


