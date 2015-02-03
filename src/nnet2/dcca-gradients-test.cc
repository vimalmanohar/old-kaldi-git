// nnet2/dcca-gradients-test.cc

// Copyright 2014  Johns Hopkins University (author:  Daniel Povey)
//           2014  Vimal Manohar

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// //  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#include "nnet2/train-nnet-dcca.h"
#include "util/common-utils.h"

namespace kaldi {
namespace nnet2 {

void CheckGradients(BaseFloat delta) {
  int32 num_rows = RandInt(512,1024),
        dim0 = RandInt(2,20),
        dim1 = RandInt(2,20), num_views = 2;
  std::vector<BaseFloat> regularizers(2,1.0);
  
  CuMatrix<BaseFloat> H0(num_rows, dim0);
  CuMatrix<BaseFloat> H1(num_rows, dim1);

  H0.SetRandn();
  H1.SetRandn();

  CuMatrix<BaseFloat> H2_0(H0);
  CuMatrix<BaseFloat> H2_1(H1);

  std::vector<CuMatrix<BaseFloat>* > H;
  std::vector<CuMatrix<BaseFloat>* > H2;
  H.reserve(2);
  H2.reserve(2);
  H.push_back(&H0);
  H.push_back(&H1);
  H2.push_back(&H2_0);
  H2.push_back(&H2_1);
  
  std::vector<CuMatrix<BaseFloat>* > derivs(num_views, 
              static_cast<CuMatrix<BaseFloat>*>(NULL));
  
  BaseFloat corr = NnetDccaTrainer::CompObjfAndGradient(H, regularizers, &derivs);

  CuMatrix<BaseFloat> D0(num_rows, dim0);
  CuMatrix<BaseFloat> D1(num_rows, dim1);

  for (int32 n = 0; n < 20; n++) {
    int32 d0 = RandInt(0, dim0);
    int32 d1 = RandInt(0, dim1);
    int32 r = RandInt(0, num_rows);

    (*H2[0])(r,d0) = (*H[0])(r,d0) + delta;
    BaseFloat corr_new = NnetDccaTrainer::CompObjfAndGradient(H2, regularizers, NULL);
    BaseFloat appx_grad = (corr_new - corr)/delta;
    if (! kaldi::ApproxEqual(appx_grad, 
          (*derivs[0])(r,d0))) {
      KALDI_ERR << "Computed gradient " << (*derivs[0])(r,d0)
        << " differs from approximated gradient " << appx_grad;
    }
    (*H2[0])(r,d0) = (*H[0])(r,d0);
    
    (*H2[1])(r,d1) = (*H[1])(r,d1) + delta;
    corr_new = NnetDccaTrainer::CompObjfAndGradient(H2, regularizers, NULL);
    appx_grad = (corr_new - corr)/delta;
    if (! kaldi::ApproxEqual(appx_grad, 
          (*derivs[1])(r,d1))) {
      KALDI_ERR << "Computed gradient " << (*derivs[1])(r,d1)
        << " differs from approximated gradient " << appx_grad;
    }
    (*H2[1])(r,d1) = (*H[1])(r,d1);
  }
}

void CheckGradientsToy(BaseFloat delta) {
  int32 num_rows = 2,
        dim0 = 1,
        dim1 = 1, num_views = 2;
  std::vector<BaseFloat> regularizers(2,1.0);
  
  CuMatrix<BaseFloat> H0(num_rows, dim0);
  CuMatrix<BaseFloat> H1(num_rows, dim1);

  H0(0,0) = 0.5;
  H0(1,0) = -0.5;
  H1(0,0) = 0.25;
  H1(1,0) = -0.25;

  CuMatrix<BaseFloat> H2_0(H0);
  CuMatrix<BaseFloat> H2_1(H1);

  std::vector<CuMatrix<BaseFloat>* > H;
  std::vector<CuMatrix<BaseFloat>* > H2;
  H.reserve(2);
  H2.reserve(2);
  H.push_back(&H0);
  H.push_back(&H1);
  H2.push_back(&H2_0);
  H2.push_back(&H2_1);
  
  std::vector<CuMatrix<BaseFloat>* > derivs(num_views, 
              static_cast<CuMatrix<BaseFloat>*>(NULL));
  
  BaseFloat corr = NnetDccaTrainer::CompObjfAndGradient(H, regularizers, &derivs);

  CuMatrix<BaseFloat> D0(num_rows, dim0);
  CuMatrix<BaseFloat> D1(num_rows, dim1);

  for (int32 n = 0; n < 20; n++) {
    int32 d0 = RandInt(0, dim0);
    int32 d1 = RandInt(0, dim1);
    int32 r = RandInt(0, num_rows);

    (*H2[0])(r,d0) = (*H[0])(r,d0) + delta;
    BaseFloat corr_new = NnetDccaTrainer::CompObjfAndGradient(H2, regularizers, NULL);
    BaseFloat appx_grad = (corr_new - corr)/delta;
    if (! kaldi::ApproxEqual(appx_grad, 
          (*derivs[0])(r,d0))) {
      KALDI_ERR << "Computed gradient " << (*derivs[0])(r,d0)
        << " differs from approximated gradient " << appx_grad;
    }
    (*H2[0])(r,d0) = (*H[0])(r,d0);
    
    (*H2[1])(r,d1) = (*H[1])(r,d1) + delta;
    corr_new = NnetDccaTrainer::CompObjfAndGradient(H2, regularizers, NULL);
    appx_grad = (corr_new - corr)/delta;
    if (! kaldi::ApproxEqual(appx_grad, 
          (*derivs[1])(r,d1))) {
      KALDI_ERR << "Computed gradient " << (*derivs[1])(r,d1)
        << " differs from approximated gradient " << appx_grad;
    }
    (*H2[1])(r,d1) = (*H[1])(r,d1);
  }
}

void CheckObj(const Matrix<BaseFloat> &mat0, const Matrix<BaseFloat> &mat1, const std::vector<BaseFloat> &regularizers) {
  CuMatrix<BaseFloat> H0(mat0);
  CuMatrix<BaseFloat> H1(mat1);

  int32 num_views = 2;

  CuMatrix<BaseFloat> H2_0(H0);
  CuMatrix<BaseFloat> H2_1(H1);

  std::vector<CuMatrix<BaseFloat>* > H;
  std::vector<CuMatrix<BaseFloat>* > H2;
  H.reserve(2);
  H2.reserve(2);
  H.push_back(&H0);
  H.push_back(&H1);
  H2.push_back(&H2_0);
  H2.push_back(&H2_1);
  
  std::vector<CuMatrix<BaseFloat>* > derivs(num_views, 
              static_cast<CuMatrix<BaseFloat>*>(NULL));
  
  BaseFloat corr = NnetDccaTrainer::CompObjfAndGradient(H, regularizers, &derivs);

  KALDI_LOG << "Objective function is " << corr;
  KALDI_LOG << "Gradient 0 is \n" << *derivs[0];
  KALDI_LOG << "Gradient 1 is \n" << *derivs[1];
}

} // namespace nnet2
} // namespace kaldi


int main(int argc, char **argv) {
  using namespace kaldi;
  using namespace kaldi::nnet2;

  const char *usage = 
    "Test DCCA gradients.\n"
    "Usage: dcca-gradients-test\n";

  BaseFloat delta = 1e-6;
  std::string mat0_rxfilename = "mat0.txt", 
    mat1_rxfilename = "mat1.txt", r = "1:1";
  ParseOptions po(usage);

  po.Register("delta", &delta, "Delta used for approximating gradient");
  po.Register("mat0-file", &mat0_rxfilename, "Matrix 0 input");
  po.Register("mat1-file", &mat1_rxfilename, "Matrix 1 input");
  po.Register("regularizer-list", &r, "Regularizer list");

  po.Read(argc, argv);

  if (po.NumArgs() != 0) {
    po.PrintUsage();
    exit(1);
  }

  Matrix<BaseFloat> mat0;
  Matrix<BaseFloat> mat1;

  ReadKaldiObject(mat0_rxfilename, &mat0);
  ReadKaldiObject(mat1_rxfilename, &mat1);

  std::vector<BaseFloat> regularizers;
  SplitStringToFloats(r, ":", false, &regularizers);
  CheckObj(mat0, mat1, regularizers);

  return 0;

  CheckGradientsToy(delta);
  for (int32 n = 0; n < 100; n++) {
    CheckGradients(delta);
  }

  return 0;
}
  
