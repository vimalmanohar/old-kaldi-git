// nnet2/nnet-multiview-example.h

// Copyright 2012  Johns Hopkins University (author: Daniel Povey)
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

#ifndef KALDI_NNET2_NNET_MULTIVIEW_EXAMPLE_H_
#define KALDI_NNET2_NNET_MULTIVIEW_EXAMPLE_H_

#include "nnet2/nnet-example.h"
#include "util/table-types.h"

namespace kaldi {
namespace nnet2 {

/// NnetMultiviewExample is the input data and corresponding labels 
/// (or labels) for one or more frames of input from atleast two views
/// Used for multiview training with DCCA objective 
struct NnetMultiviewExample {
  /// Number of views of data (Expecting atleast 2)
  int32 num_views;

  /// NnetExamples corresponding to the different views
  std::vector<NnetExample> views;

  void Write(std::ostream &os, bool binary) const;
  void Read(std::istream &is, bool binary);

  NnetMultiviewExample() { }

  NnetMultiviewExample(const std::vector<NnetExample> &eg_views);
  NnetMultiviewExample(const NnetMultiviewExample &eg);
  NnetMultiviewExample(const NnetMultiviewExample &input,
                       int32 start_frame,
                       int32 new_num_frames,
                       int32 new_left_context,
                       int32 new_right_context);
};

typedef TableWriter<KaldiObjectHolder<NnetMultiviewExample > >      
  NnetMultiviewExampleWriter;
typedef SequentialTableReader<KaldiObjectHolder<NnetMultiviewExample > > 
  SequentialNnetMultiviewExampleReader;
typedef RandomAccessTableReader<KaldiObjectHolder<NnetMultiviewExample > > 
  RandomAccessNnetMultiviewExampleReader;


}
} // namespace

#endif // KALDI_NNET2_NNET_MULTIVIEW_EXAMPLE_H_

