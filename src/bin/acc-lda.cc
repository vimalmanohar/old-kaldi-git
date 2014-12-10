// bin/acc-lda.cc

// Copyright 2009-2011  Microsoft Corporation, Go-Vivace Inc.
//                2014  Guoguo Chen

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
#include "hmm/transition-model.h"
#include "hmm/posterior.h"
#include "transform/lda-estimate.h"

/** @brief Accumulate LDA statistics based on pdf-ids. Inputs are the
source models, that serve as the input (and may potentially contain
the current transformation), the un-transformed features and state
posterior probabilities */
int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
  try {
    typedef unordered_map<std::string, std::string, StringHasher> string_map;
    const char *usage =
        "Accumulate LDA statistics based on pdf-ids.\n"
        "Usage:  acc-lda [options] <transition-gmm/model> <features-rspecifier> <posteriors-rspecifier> <lda-acc-out>\n"
        "Typical usage:\n"
        " ali-to-post ark:1.ali ark:- | lda-acc 1.mdl \"ark:splice-feats scp:train.scp|\"  ark:- ldaacc.1\n";

    bool binary = true;
    BaseFloat rand_prune = 0.0;
    std::string utt2uniq;
    
    ParseOptions po(usage);
    po.Register("binary", &binary, "Write accumulators in binary mode.");
    po.Register("rand-prune", &rand_prune, "Randomized pruning threshold for posteriors");
    po.Register("utt2uniq", &utt2uniq, "Used to map the utt_id of "
                "artificial utterances in features to utt_id of 'real' "
                "utterances in alignment.");
    po.Read(argc, argv);

    if (po.NumArgs() != 4) {
      po.PrintUsage();
      exit(1);
    }
    
    string_map utt2uniq_map;
    if (utt2uniq != "") {
      bool binary;
      Input ki(utt2uniq, &binary);
      KALDI_ASSERT(!binary);
      std::string line;
      while (std::getline(ki.Stream(), line)) {
        std::vector<std::string> split_line;
        SplitStringToVector(line, " \t\r", true, &split_line);
        if(split_line.size() != 2) {
          KALDI_ERR << "Unable to parse line \"" << line << "\" encountered in input in " << utt2uniq;
        }
        utt2uniq_map[split_line[0]] = split_line[1];
      }
    }

    std::string model_rxfilename = po.GetArg(1);
    std::string features_rspecifier = po.GetArg(2);
    std::string posteriors_rspecifier = po.GetArg(3);
    std::string acc_wxfilename = po.GetArg(4);

    TransitionModel trans_model;
    {
      bool binary_read;
      Input ki(model_rxfilename, &binary_read);
      trans_model.Read(ki.Stream(), binary_read);
      // discard rest of file.
    }

    LdaEstimate lda;

    SequentialBaseFloatMatrixReader feature_reader(features_rspecifier);
    RandomAccessPosteriorReader posterior_reader(posteriors_rspecifier);

    int32 num_done = 0, num_fail = 0;
    for (;!feature_reader.Done(); feature_reader.Next()) {
      std::string utt = feature_reader.Key(), uniq_utt = utt;
      if (utt2uniq != "") {
        string_map::const_iterator got = utt2uniq_map.find(utt);
        if (got == utt2uniq_map.end()) 
          KALDI_ERR << "Key " << utt << " not found in " << utt2uniq;
        uniq_utt = utt2uniq_map[utt];
      }

      if (!posterior_reader.HasKey(uniq_utt)) {
        KALDI_WARN << "No posteriors for utterance " << uniq_utt;
        num_fail++;
        continue;
      }
      const Posterior &post (posterior_reader.Value(uniq_utt));
      const Matrix<BaseFloat> &feats(feature_reader.Value());

      if (lda.Dim() == 0)
        lda.Init(trans_model.NumPdfs(), feats.NumCols());

      if (feats.NumRows() != static_cast<int32>(post.size())) {
        KALDI_WARN << "Posterior vs. feats size mismatch "
                   << feats.NumRows() << " vs. " <<post.size();
        num_fail++;
        continue;
      }
      if (lda.Dim() != 0 && lda.Dim() != feats.NumCols()) {
        KALDI_WARN << "Feature dimension mismatch " << lda.Dim()
                   << " vs. " << feats.NumCols();
        num_fail++;
        continue;
      }

      Posterior pdf_post;
      ConvertPosteriorToPdfs(trans_model, post, &pdf_post);
      for (int32 i = 0; i < feats.NumRows(); i++) {
        SubVector<BaseFloat> feat(feats, i);
        for (size_t j = 0; j < pdf_post[i].size(); j++) {
          int32 pdf_id = pdf_post[i][j].first;
          BaseFloat weight = RandPrune(pdf_post[i][j].second, rand_prune);
          if (weight != 0.0) {
            lda.Accumulate(feat, pdf_id, weight);
          }
        }
      }
      num_done++;
      if (num_done % 100 == 0)
        KALDI_LOG << "Done " << num_done << " utterances.";
    }

    KALDI_LOG << "Done " << num_done << " files, failed for "
              << num_fail;

    Output ko(acc_wxfilename, binary);
    lda.Write(ko.Stream(), binary);
    KALDI_LOG << "Written statistics.";
    return (num_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


