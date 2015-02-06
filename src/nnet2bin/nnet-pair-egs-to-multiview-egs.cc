// nnet2bin/nnet-pair-egs-to-multiview-egs.cc

// Copyright 2015 Vimal Manohar

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
#include "nnet2/nnet-multiview-example.h"
#include "nnet2/nnet-example-functions.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet2;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    const char *usage =
        "Get frame-by-frame multiview examples of data for "
        "neural network training by combining two nnet examples.\n"
        "This is useful for frame-pairs based training like in "       
        "zero-resource modeling, but using an objective like CCA."
        "This combines examples sequentially from the two given "
        "NnetExamples, which just contain frames of the same class or "
        "cluster. This is different from nnet-get-multiview-egs, which "
        "creates multiview egs from two different datasets"
        "Usage:  nnet-pair-egs-to-multiview-egs [options] "
        "<egs-rspecifier-1> <egs-rspecifier-2> ... "
        "<egs-rspecifier-N> <multiview-egs-wspecifier>\n"
        "\n"
        "nnet-pair-egs-to-multiview-egs ark:egs.1.ark "
        "\"ark:nnet-shuffle-egs ark:egs.1.ark ark:- |\" "
        "ark:multiview_egs\n";

    int32 left_context = 0, right_context = 0,
        num_frames = 1, const_feat_dim = 0;
    bool ignore_labels = false;
    
    ParseOptions po(usage);
    po.Register("left-context", &left_context, "Number of frames of left "
                "context the neural net requires.");
    po.Register("right-context", &right_context, "Number of frames of right "
                "context the neural net requires.");
    po.Register("num-frames", &num_frames, "Number of frames with labels "
                "that each example contains.");
    po.Register("const-feat-dim", &const_feat_dim, "If specified, the last "
                "const-feat-dim dimensions of the feature input are treated as "
                "constant over the context window (so are not spliced)");
    po.Register("ignore-labels", &ignore_labels, "Ignores labels while "
                "creating examples. This might be useful for unsupervised "
                "methods like DCCA");
    
    po.Read(argc, argv);

    if (po.NumArgs() < 3 || po.NumArgs() % 2 == 0) {
      po.PrintUsage();
      exit(1);
    }

    int32 num_views = (po.NumArgs() - 1);

    std::string examples_wspecifier = po.GetArg(po.NumArgs());

    std::vector<SequentialNnetExampleReader*> egs_readers(
        num_views, static_cast<SequentialNnetExampleReader*>(NULL));

    for (int32 i = 0; i < num_views; i++) {
      egs_readers[i] = new SequentialNnetExampleReader(po.GetArg(i+1));
    }

    NnetMultiviewExampleWriter example_writer(examples_wspecifier);

    int32 num_done = 0, num_err = 0;
    int64 num_frames_written = 0, num_egs_written = 0;

    bool done_flag = false;
    for (; !done_flag; num_frames_written++, num_egs_written++) {
      NnetMultiviewExample eg;
      eg.num_views = num_views;
      eg.views.resize(num_views);

      std::ostringstream os;
      os << egs_readers[0]->Key();

      for (int32 i = 0; i < num_views; i++) {
        eg.views[i] = egs_readers[i]->Value();
        //if (eg.views[i].num_frames != 1) {
        //  KALDI_ERR << "Currently, this program does not support "           
        //            << "num_frames != 1"
        //}
        egs_readers[i]->Next();
        if (egs_readers[i]->Done()) done_flag = true;
      }

      example_writer.Write(std::string(os.str()), eg);
    }
      
    KALDI_LOG << "Wrote " << num_egs_written << " examples "
              << " with " << num_frames_written << " frames.";
    return (num_egs_written == 0 ? 1 : 0);
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}




