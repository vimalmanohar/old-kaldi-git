// nnet2/train-nnet-dcca.cc

// Copyright 2012  Johns Hopkins University (author: Daniel Povey)
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

#include "nnet2/train-nnet-dcca.h"
#include "nnet2/nnet-multiview-example.h"
#include "util/text-utils.h"

namespace kaldi {
namespace nnet2 {

NnetDccaTrainer::NnetDccaTrainer(
    const NnetDccaTrainerConfig &config,
    std::vector<Nnet*> nnet_views):
    config_(config), nnet_views_(nnet_views),
    avg_correlation_this_phase_(0.0), count_this_phase_(0.0),
    avg_correlation_(0.0), count_total_(0.0) {
  num_phases_ = 0;
  bool first_time = true;
  BeginNewPhase(first_time);
  SplitStringToFloats(config_.r, ":", false, &regularizers_);
  buffers_.resize(nnet_views.size());
  for (size_t i = 0; i < nnet_views.size(); i++) {
    buffers_[i].reserve(config.minibatch_size);
  }
}

void NnetDccaTrainer::TrainOnExample(const NnetMultiviewExample &value) {
  KALDI_ASSERT(buffers_.size() == value.num_views); // Number of views
  for (size_t i = 0; i < value.num_views; i++) {
    buffers_[i].push_back(value.views[i]);
  }
  // Although we check for only one view, all views would be of the same size
  // here
  if (static_cast<int32>(buffers_[0].size()) == config_.minibatch_size)
    TrainOneMinibatch();
}

void NnetDccaTrainer::TrainOneMinibatch() {
  KALDI_ASSERT(!buffers_[0].empty());
  int32 num_views = nnet_views_.size();

  updater_views_.reserve(num_views);
  std::vector<CuMatrix<BaseFloat>* > H(num_views,
        static_cast<CuMatrix<BaseFloat>* >(NULL));
  std::vector<CuMatrix<BaseFloat>* > tmp_derivs(num_views, 
        static_cast<CuMatrix<BaseFloat>* >(NULL));
        
  for (int32 i = 0; i < nnet_views_.size(); i++) {
    updater_views_.push_back(new NnetUpdater(*(nnet_views_[i]), config_.updater_config, nnet_views_[i]));
    updater_views_[i]->FormatInput(buffers_[i]);
    updater_views_[i]->Propagate();
    H[i] = new CuMatrix<BaseFloat>;
    updater_views_[i]->GetOutput(H[i]);
    
    KALDI_ASSERT(H[i]->NumRows() == config_.minibatch_size);
  }

  if (num_views != 2) {
    KALDI_ERR << "DCCA currently supports only two views.";
  }

  BaseFloat corr = CompObjfAndGradient(H, regularizers_, &tmp_derivs);
  updater_views_[0]->Backprop(tmp_derivs[0]);
  updater_views_[1]->Backprop(tmp_derivs[1]);

  // Accumulate number of frames in this phase 
  count_this_phase_ += buffers_[0].size(); 
  
  avg_correlation_this_phase_ += corr;
  
  for (int32 i = 0; i < nnet_views_.size(); i++) {
    buffers_[i].clear();
  }

  minibatches_seen_this_phase_++;
  if (minibatches_seen_this_phase_ == config_.minibatches_per_phase) {
    avg_correlation_this_phase_ /= minibatches_seen_this_phase_;
    bool first_time = false;
    BeginNewPhase(first_time);
  }
} 

void NnetDccaTrainer::BeginNewPhase(bool first_time) {
  if (!first_time) 
    KALDI_LOG << "Average correlation between the two views is "
              << avg_correlation_this_phase_ << " over " 
              << count_this_phase_ << " frames, during this phase";
  avg_correlation_ += avg_correlation_this_phase_;
  count_total_ += count_this_phase_;
  avg_correlation_this_phase_ = 0.0;
  count_this_phase_ = 0.0;
  minibatches_seen_this_phase_ = 0;
  num_phases_++;
}

NnetDccaTrainer::~NnetDccaTrainer() {
  if (!buffers_[0].empty()) {
    KALDI_LOG << "Ignoring partial minibatch of size "
              << buffers_[0].size();
    //TrainOneMinibatch();
    //if (minibatches_seen_this_phase_ != 0) {
    //  bool first_time = false;
    //  BeginNewPhase(first_time);
    //}
  }

  KALDI_LOG << "Did backprop on " << count_total_ 
            << " examples, average correlation between the two views is "
            << (avg_correlation_ / num_phases_);
  KALDI_LOG << "[this line is to be parsed by a script:] average-correlation="
            << (avg_correlation_ / num_phases_);
}

BaseFloat NnetDccaTrainer::CompObjfAndGradient(
            const std::vector<CuMatrix<BaseFloat>* > &H,
            const std::vector<BaseFloat> &regularizers, 
            std::vector<CuMatrix<BaseFloat>* > *tmp_derivs) {
  int32 num_views = H.size();
  int32 num_samples = H[0]->NumRows();
  
  std::vector<CuTpMatrix<BaseFloat>* > 
        R_hat(num_views,
        static_cast<CuTpMatrix<BaseFloat>* >(NULL));
  
  std::vector<CuMatrix<BaseFloat>* > H_bar(num_views,
        static_cast<CuMatrix<BaseFloat>* >(NULL));

  for (int32 i = 0; i < num_views; i++) {
    // Center the data matrix
    H_bar[i] = new CuMatrix<BaseFloat>(*H[i]);
    {
      CuVector<BaseFloat> mu(H[i]->NumCols());
      mu.AddRowSumMat(1.0, *(H[i]), 0.0);
      H_bar[i]->AddVecToRows(-1.0/num_samples, mu);
    }
    
    // Compute the covariance matrix
    CuSpMatrix<BaseFloat> Sigma_hat(H_bar[i]->NumCols());
    Sigma_hat.AddMat2(1.0/(num_samples-1), 
                                   *(H_bar[i]), kTrans, 0.0);
    Sigma_hat.AddToDiag(regularizers[i]);
   
    R_hat[i] = new CuTpMatrix<BaseFloat>(H_bar[i]->NumCols());
    R_hat[i]->Cholesky(Sigma_hat);
    R_hat[i]->Invert();
  }
  
  KALDI_VLOG(4) << "H_bar0: \n" << *H_bar[0];
  KALDI_VLOG(4) << "H_bar1: \n" << *H_bar[1];
  
  KALDI_VLOG(4) << "R_hat0: \n" << *R_hat[0];
  KALDI_VLOG(4) << "R_hat1: \n" << *R_hat[1];
  
  CuMatrix<BaseFloat> Sigma_hat_01(H_bar[0]->NumCols(),
                                   H_bar[1]->NumCols());
  Sigma_hat_01.AddMatMat(1.0/(num_samples-1), 
      *(H_bar[0]), kTrans, *(H_bar[1]), kNoTrans, 0.0);
  
  KALDI_VLOG(4) << "Sigma_hat_01: \n" << Sigma_hat_01;

  CuMatrix<BaseFloat> T(H_bar[0]->NumCols(), H_bar[1]->NumCols());
  {
    CuMatrix<BaseFloat> temp(H_bar[0]->NumCols(), H_bar[1]->NumCols());
    temp.AddMatTp(1.0, Sigma_hat_01, kNoTrans, 
                  *(R_hat[1]), kNoTrans, 0.0);
    T.AddTpMat(1.0, *(R_hat[0]), kNoTrans, temp, kNoTrans, 0.0);
  }
  
  KALDI_VLOG(4) << "T: \n" << T;

  SpMatrix<BaseFloat> Delta_0(H_bar[0]->NumCols());
  SpMatrix<BaseFloat> Delta_1(H_bar[1]->NumCols());
  Matrix<BaseFloat> Delta_01(H_bar[0]->NumCols(), H_bar[1]->NumCols());

  double corr = 0;
  {
    int32 min_rc = std::min(T.NumRows(), T.NumCols());
    Vector<BaseFloat> s(min_rc);
    Matrix<BaseFloat> U(T.NumRows(), min_rc);
    Matrix<BaseFloat> Vt(min_rc, T.NumCols());
    
    Matrix<BaseFloat>(T).Svd(&s, &U, &Vt);
    corr = s.Norm(2);

    if (tmp_derivs == NULL) {
      return corr;
    }

    // Compute Delta_0 
    Matrix<BaseFloat> RtU_0(Delta_0.NumRows(), Delta_0.NumCols());
    RtU_0.AddTpMat(1.0, TpMatrix<BaseFloat>(*(R_hat[0])), kTrans, U, kNoTrans, 0.0);
    Delta_0.AddMat2Vec(-0.5, RtU_0, kNoTrans, s, 0.0);

    // Compute Delta_1
    Matrix<BaseFloat> VtRt_1(Delta_1.NumRows(), Delta_1.NumCols());
    VtRt_1.AddMatTp(1.0, Vt, kNoTrans, TpMatrix<BaseFloat>(*(R_hat[0])), kTrans, 0.0);
    Delta_1.AddMat2Vec(-0.5, VtRt_1, kNoTrans, s, 0.0);

    // Compute Delta_01
    Delta_01.AddMatMat(1.0, RtU_0, kNoTrans, VtRt_1, kNoTrans, 0.0);
  }

  KALDI_VLOG(4) << "Delta0: \n" << Delta_0;
  KALDI_VLOG(4) << "Delta1: \n" << Delta_1;
  KALDI_VLOG(4) << "Delta01: \n" << Delta_01;
  
  (*tmp_derivs)[0] = new CuMatrix<BaseFloat>(H_bar[0]->NumRows(), 
                                          H_bar[0]->NumCols());
  (*tmp_derivs)[1] = new CuMatrix<BaseFloat>(H_bar[1]->NumRows(), 
                                          H_bar[1]->NumCols());

  {
    CuMatrix<BaseFloat> &tmp_deriv(*(*tmp_derivs)[0]);
    tmp_deriv.AddMatSp(2.0/(num_samples-1), *(H_bar[0]), kNoTrans,
                       CuSpMatrix<BaseFloat>(Delta_0), 0.0);
    tmp_deriv.AddMatMat(1.0/(num_samples-1), *(H_bar[1]), kNoTrans,
                       CuMatrix<BaseFloat>(Delta_01), kTrans, 1.0);
  }

  {
    CuMatrix<BaseFloat> &tmp_deriv(*(*tmp_derivs)[1]);
    tmp_deriv.AddMatSp(2.0/(num_samples-1), *(H_bar[1]), kNoTrans,
                       CuMatrix<BaseFloat>(Delta_1), 0.0);
    tmp_deriv.AddMatMat(1.0/(num_samples-1), *(H_bar[0]), kNoTrans,
                       CuMatrix<BaseFloat>(Delta_01), kNoTrans, 1.0);
  }

  return corr;
}

BaseFloat NnetDccaTrainer::ComputeDccaObjf(
                          const std::vector<Nnet*> nnet_views,
                          const std::vector<std::vector<NnetExample> > &buffers,
                          const std::vector<BaseFloat> &regularizers,
                          NnetUpdaterConfig updater_config) {
  int32 num_views = nnet_views.size();
  
  std::vector<NnetUpdater*> updater_views;
  updater_views.reserve(num_views);

  std::vector<CuMatrix<BaseFloat>* > H(num_views,
        static_cast<CuMatrix<BaseFloat>* >(NULL));
        
  for (int32 i = 0; i < num_views; i++) {
    updater_views.push_back(new NnetUpdater(*(nnet_views[i]), updater_config, static_cast<Nnet*>(NULL)));
    updater_views[i]->FormatInput(buffers[i]);
    updater_views[i]->Propagate();
    H[i] = new CuMatrix<BaseFloat>;
    updater_views[i]->GetOutput(H[i]);
  }

  if (num_views != 2) {
    KALDI_ERR << "DCCA currently supports only two views.";
  }

  return CompObjfAndGradient(H, regularizers, NULL);
}

} // namespace nnet2
} // namespace kaldi
