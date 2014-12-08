// nnet/nnet-multiview-example.cc

// Copyright 2012-2013  Johns Hopkins University (author: Daniel Povey)
// Copyright 2014       Vimal Manohar

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

#include "nnet2/nnet-example.h"
#include "nnet2/nnet-multiview-example.h"

namespace kaldi {
namespace nnet2 {

void NnetMultiviewExample::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<NnetMultiviewExample>");
  WriteToken(os, binary, "<NumViews>");
  WriteBasicType(os, binary, num_views);
  for (int32 i = 0; i < num_views; i++) {
    views[i].Write(os, binary);
  }
}

void NnetMultiviewExample::Read(std::istream &is, bool binary) {
  ExpectToken(is, binary, "<NnetMultiviewExample>");
  ExpectToken(is, binary, "<NumViews>");
  ReadBasicType(is, binary, &num_views);

  NnetExample eg;
  views.reserve(num_views);
  for (int32 i = 0; i < num_views; i++) {
    eg.Read(is, binary);
    views.push_back(eg);
  }
}

NnetMultiviewExample::NnetMultiviewExample(const std::vector<NnetExample> &eg_views) {
  num_views = eg_views.size();
  views.reserve(num_views);
  for (int32 i = 0; i < num_views; i++) {
    views.push_back(eg_views[i]);
  }
}

NnetMultiviewExample::NnetMultiviewExample(const NnetMultiviewExample &eg) {
  num_views = eg.num_views;
  views.reserve(num_views);
  for (int32 i = 0; i < num_views; i++) {
    views.push_back(eg.views[i]);
  }
}

NnetMultiviewExample::NnetMultiviewExample(
                            const NnetMultiviewExample &input,
                            int32 start_frame,
                            int32 new_num_frames,
                            int32 new_left_context,
                            int32 new_right_context) {
  num_views = input.num_views;

  views.reserve(num_views);
  for (int32 i = 0; i < num_views; i++) {
    views.push_back(NnetExample(input.views[i], start_frame, 
                                new_num_frames, new_left_context, 
                                new_right_context));
  }
}

} // namespace nnet2
} // namespace kaldi
