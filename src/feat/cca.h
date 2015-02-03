// feat/cca.h

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

#ifndef KALDI_FEAT_CCA_H
#define KALDI_FEAT_CCA_H_

namespace kaldi {

class CcaStats {
  Vector<BaseFloat> x1_;
  Vector<BaseFloat> y1_;
  SpMatrix<BaseFloat> x2_;
  SpMatrix<BaseFloat> y2_;
  Matrix<BaseFloat> xy_;
  int32 num_feats_;
  int32 dim_;

  public:
  inline int32 NumFeats() {return num_feats_;}
  inline int32 Dim() {return dim_;}

  const Vector<BaseFloat> &x1() { return x1_; }
  const Vector<BaseFloat> &y1() { return y1_; }
  const SpMatrix<BaseFloat> &x2() { return x2_; }
  const SpMatrix<BaseFloat> &y2() { return y2_; }
  const Matrix<BaseFloat> &xy() { return xy_; }
  
  void Resize(int32 dim) {
    dim_ = dim;
    num_feats_ = 0;
    x1_.Resize(dim);
    y1_.Resize(dim);
    x2_.Resize(dim);
    y2_.Resize(dim);
    xy_.Resize(dim,dim);
  }

  void Sum(CcaStats &other) {
    x1_.AddVec(1.0, other.x1());
    y1_.AddVec(1.0, other.y1());
    xy_.AddMat(1.0, other.xy(), kNoTrans);
    x2_.AddSp(1.0, other.x2());
    y2_.AddSp(1.0, other.y2());
    num_feats_ += other.NumFeats();
  }

  void Accumulate(const Matrix<BaseFloat> &X, const Matrix<BaseFloat> &Y) {
    x1_.AddRowSumMat(1.0, X, 1.0);
    y1_.AddRowSumMat(1.0, Y, 1.0);
    xy_.AddMatMat(1.0, X, kTrans, Y, kNoTrans, 1.0);
    x2_.AddMat2(1.0, X, kTrans, 1.0);
    y2_.AddMat2(1.0, Y, kTrans, 1.0);
    num_feats_ += X.NumRows();
  }

  void Read(std::istream &in_stream, bool binary) {
    ExpectToken(in_stream, binary, "<NumFeats>");
    ReadBasicType(in_stream, binary, &num_feats_);
    ExpectToken(in_stream, binary, "<Dim>");
    ReadBasicType(in_stream, binary, &dim_);

    x1_.Resize(dim_);
    y1_.Resize(dim_);
    x2_.Resize(dim_);
    y2_.Resize(dim_);
    xy_.Resize(dim_,dim_);

    ExpectToken(in_stream, binary, "<x1>");
    x1_.Read(in_stream, binary);
    ExpectToken(in_stream, binary, "<x2>");
    x2_.Read(in_stream, binary);
    ExpectToken(in_stream, binary, "<y1>");
    y1_.Read(in_stream, binary);
    ExpectToken(in_stream, binary, "<y2>");
    y2_.Read(in_stream, binary);
    ExpectToken(in_stream, binary, "<xy>");
    xy_.Read(in_stream, binary);
  }

  void Write(std::ostream &out_stream, bool binary) const {
    WriteToken(out_stream, binary, "<NumFeats>");
    WriteBasicType(out_stream, binary, num_feats_);
    WriteToken(out_stream, binary, "<Dim>");
    WriteBasicType(out_stream, binary, dim_);

    WriteToken(out_stream, binary, "<x1>");
    x1_.Write(out_stream, binary);
    WriteToken(out_stream, binary, "<x2>");
    x2_.Write(out_stream, binary);
    WriteToken(out_stream, binary, "<y1>");
    y1_.Write(out_stream, binary);
    WriteToken(out_stream, binary, "<y2>");
    y2_.Write(out_stream, binary);
    WriteToken(out_stream, binary, "<xy>");
    xy_.Write(out_stream, binary);
  }
};

} // namespace kaldi

#endif // KALDI_FEAT_CCA_H
