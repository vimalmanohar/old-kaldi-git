// nnet2bin/nnet-get-multiview-egs.cc

// Copyright 2012-2014  Johns Hopkins University (author:  Daniel Povey)
//                2014  Vimal Manohar

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

#include <sstream>

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "hmm/transition-model.h"
#include "nnet2/nnet-multiview-example.h"
#include "nnet2/nnet-example-functions.h"

namespace kaldi {
namespace nnet2 {


static void ProcessFiles(const std::vector<const Matrix<BaseFloat>*> &feats_views,
                         const std::vector<const Posterior*> &pdf_post_views,
                         const std::string &utt_id,
                         bool ignore_labels,
                         int32 left_context,
                         int32 right_context,
                         int32 num_frames,
                         int32 const_feat_dim,
                         int64 *num_frames_written,
                         int64 *num_egs_written,
                         NnetMultiviewExampleWriter *example_writer) {
  int32 num_views = feats_views.size();
  KALDI_ASSERT(num_views >= 2);
  
  const Matrix<BaseFloat>* feats = feats_views[0];

  if (!ignore_labels) {
    const Posterior* pdf_post = pdf_post_views[0];
    KALDI_ASSERT(feats->NumRows() == static_cast<int32>(pdf_post->size()));
  }

  int32 feat_dim = feats->NumCols();
  KALDI_ASSERT(const_feat_dim < feat_dim);
  KALDI_ASSERT(num_frames > 0);
  int32 basic_feat_dim = feat_dim - const_feat_dim;

  for (int32 t = 0; t < feats->NumRows(); t += num_frames) {
    int32 this_num_frames = std::min(num_frames,
                                     feats->NumRows() - t);

    int32 tot_frames = left_context + this_num_frames + right_context;

    NnetMultiviewExample multiview_eg;
    multiview_eg.num_views = num_views;
    multiview_eg.views.resize(num_views);

    for (int32 i = 0; i < num_views; i++) {
      const Matrix<BaseFloat>* feats = feats_views[i];

      NnetExample &eg = multiview_eg.views[i];
      Matrix<BaseFloat> input_frames(tot_frames, basic_feat_dim);
      eg.left_context = left_context;
      eg.spk_info.Resize(const_feat_dim);

      // Set up "input_frames".
      for (int32 j = -left_context; j < this_num_frames + right_context; j++) {
        int32 t2 = j + t;
        if (t2 < 0) t2 = 0;
        if (t2 >= feats->NumRows()) t2 = feats->NumRows() - 1;
        SubVector<BaseFloat> src(feats->Row(t2), 0, basic_feat_dim),
          dest(input_frames, j + left_context);
        dest.CopyFromVec(src);
        if (const_feat_dim > 0) {
          SubVector<BaseFloat> src(feats->Row(t2), basic_feat_dim, const_feat_dim);
          // set eg.spk_info to the average of the corresponding dimensions of
          // the input, taken over the frames whose features we store in the eg.
          eg.spk_info.AddVec(1.0 / tot_frames, src);
        }
      }
      eg.labels.resize(this_num_frames);
      if (!ignore_labels) {
        const Posterior* pdf_post = pdf_post_views[i];
        for (int32 j = 0; j < this_num_frames; j++)
          eg.labels[j] = (*pdf_post)[t + j];
      } else {
        for (int32 j = 0; j < this_num_frames; j++)
          eg.labels[j] = std::vector<std::pair<int32, BaseFloat> > (1, 
                                      std::make_pair(0, 1.0));
      }

      eg.input_frames = input_frames;  // Copy to CompressedMatrix.
    }
    
    std::ostringstream os;
    os << utt_id << "-" << t;

    std::string key = os.str(); // key is <utt_id>-<frame_id>

    *num_frames_written += this_num_frames;
    *num_egs_written += 1;

    example_writer->Write(key, multiview_eg);
  }
}

} // namespace nnet2
} // namespace kaldi

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet2;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    const char *usage =
        "Get frame-by-frame multiview examples of data for "
        "neural network training.\n"
        "Essentially this is a format change from features and posteriors\n"
        "into a special frame-by-frame format.\n"
        "This program is similar to nnet-get-egs but works with "
        "data from two different views\n"
        "The pdf-post-rspecifiers can be /dev/null "
        "if --ignore-labels=true is specified.\n"
        "Usage:  nnet-get-multiview-egs [options] "
        "<features-rspecifier-1> <pdf-post-rspecifier-1> "
        "<features-rspecifier-2> <pdf-post-rspecifier-2> ... "
        "<features-rspecifier-N> <pdf-post-rspecifier-N> "
        "<training-examples-out>\n"
        "\n"
        "An example [where $feats_view1 and $feats_view2 expand to the "
        "actual features]:\n"
        "nnet-get-multiview-egs --ignore-labels=true --left-context=8 "
        "--right-context=8 \"$feats\" ark:/dev/null \\\n"
        "\"$feats\" ark:/dev/null \"$feats_view2\" ark:/dev/null ark:- \n";
    
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

    if (po.NumArgs() < 5 || po.NumArgs() % 2 == 0) {
      po.PrintUsage();
      exit(1);
    }

    int32 num_views = (po.NumArgs() - 1) / 2;

    std::vector<std::string> feature_rspecifiers, 
                             pdf_post_rspecifiers;
    std::string examples_wspecifier = po.GetArg(po.NumArgs());

    SequentialBaseFloatMatrixReader feats_reader0(po.GetArg(1));
    std::vector<RandomAccessBaseFloatMatrixReader*> feats_readers(
        num_views,
        static_cast<RandomAccessBaseFloatMatrixReader*>(NULL));
    feature_rspecifiers.push_back(po.GetArg(1));
    
    std::vector<RandomAccessPosteriorReader*> pdf_post_readers(
        num_views,
        static_cast<RandomAccessPosteriorReader*>(NULL));
    pdf_post_readers[0] = new RandomAccessPosteriorReader(po.GetArg(2));
    pdf_post_rspecifiers.push_back(po.GetArg(2));

    for (int32 i = 1; i < num_views; i++) {
      feature_rspecifiers.push_back(po.GetArg(2*i+1));
      pdf_post_rspecifiers.push_back(po.GetArg(2*i+2));
      feats_readers[i] = new RandomAccessBaseFloatMatrixReader(
            feature_rspecifiers[i]);
      pdf_post_readers[i] = new RandomAccessPosteriorReader(
            pdf_post_rspecifiers[i]);
    }

    NnetMultiviewExampleWriter example_writer(examples_wspecifier);

    int32 num_done = 0, num_err = 0;
    int64 num_frames_written = 0, num_egs_written = 0;
    
    for (; !feats_reader0.Done(); feats_reader0.Next()) {
      std::string key = feats_reader0.Key();
      if (!ignore_labels && !pdf_post_readers[0]->HasKey(key)) {
        KALDI_WARN << "No pdf-level posterior for key " << key;
        num_err++;
        continue;
      }
      
      bool all_views_present = true;
      for (int32 i = 1; i < num_views; i++) {
        if (!ignore_labels && !pdf_post_readers[i]->HasKey(key)) {
          KALDI_WARN << "No pdf-level posterior for key " << key;
          num_err++;
          all_views_present = false;
          break;
        }
        if (!feats_readers[i]->HasKey(key)) {
          KALDI_WARN << "No features for key " << key << " in " << feature_rspecifiers[i];
          num_err++;
          all_views_present = false;
          break;
        }
      } 
      if (!all_views_present) continue;
      
      std::vector<const Posterior*> pdf_post_views;
      std::vector<const Matrix<BaseFloat>*> feats_views;
      
      {
        const Matrix<BaseFloat> *feats = &(feats_reader0.Value());

        if (!ignore_labels) {
          const Posterior *pdf_post = &(pdf_post_readers[0]->Value(key));

          if (pdf_post->size() != feats->NumRows()) {
            KALDI_WARN << "Posterior has wrong size " << pdf_post->size()
              << " versus " << feats->NumRows()
              << " for view 1";
            num_err++;
            continue;
          }
          pdf_post_views.push_back(pdf_post);
        }
        feats_views.push_back(feats);
      }

      {
        all_views_present = true;
        for (int32 i = 1; i < num_views; i++) {
          const Matrix<BaseFloat> *feats = 
            &(feats_readers[i]->Value(key));
          
          if (feats->NumRows() != feats_views[0]->NumRows()) {
            KALDI_WARN << "Features has wrong size " << feats->NumRows()
              << " versus " << feats_views[0]->NumRows()
              << " for view " << i;
            num_err++;
            all_views_present = false;
            break;
          }

          if (!ignore_labels) {
            const Posterior *pdf_post = 
              &(pdf_post_readers[i]->Value(key));
        
            if (pdf_post->size() != feats_views[0]->NumRows()) {
              KALDI_WARN << "Posterior has wrong size " << pdf_post->size()
                << " versus " << feats_views[0]->NumRows()
                << " for view " << i;
              num_err++;
              all_views_present = false;
              break;
            }
            pdf_post_views.push_back(pdf_post);
          }
          
          feats_views.push_back(feats);
        }
        if (!all_views_present) continue;

        ProcessFiles(feats_views, pdf_post_views, key, ignore_labels, 
                     left_context, right_context, num_frames,
                     const_feat_dim, &num_frames_written, 
                     &num_egs_written, &example_writer);
        num_done++;
      }
    }

    KALDI_LOG << "Finished generating examples, "
              << "successfully processed " << num_done
              << " feature files, wrote " << num_egs_written << " examples, "
              << " with " << num_frames_written << " egs in total; "
              << num_err << " files had errors.";
    return (num_egs_written == 0 ? 1 : 0);
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}

