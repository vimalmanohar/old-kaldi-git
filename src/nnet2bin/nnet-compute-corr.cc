// nnet2bin/nnet-compute-corr.cc

// Copyright 2012  Johns Hopkins University (author:  Daniel Povey)
//           2014  Vimal Manohar

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
#include "nnet2/train-nnet-dcca.h"
#include "nnet2/am-nnet.h"


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet2;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    const char *usage =
        "Computes and prints the average correlation of the given data. \n"
        "The input of this is the output of e.g. nnet-get-multiview-egs\n"
        "By default reads model file (.mdl) but with --raw=true,\n"
        "reads/writes raw-nnet.\n"
        "\n"
        "Usage:  nnet-compute-corr [options] <model-in> <training-examples-in>\n"
        "e.g.: nnet-compute-corr 1.nnet ark:valid.egs\n";
 
    NnetUpdaterConfig updater_config;
    bool raw = false;
    NnetDccaTrainerConfig train_config;

    ParseOptions po(usage);
    
    updater_config.Register(&po);
    po.Register("raw", &raw,
                "If true, read/write raw neural net rather than .mdl");
    train_config.Register(&po);

    po.Read(argc, argv);
   
    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }
    
    std::vector<std::string> nnet_rxfilenames;
    std::string examples_rspecifier = po.GetArg(3);
    nnet_rxfilenames.push_back(po.GetArg(1));
    nnet_rxfilenames.push_back(po.GetArg(2));

    std::vector<AmNnet> am_nnets(2);
    std::vector<Nnet*> nnets(2);

    TransitionModel trans_model;
    if (!raw) {
      for (int32 i = 0; i < 2; i++) {
        bool binary_read;
        Input ki(nnet_rxfilenames[i], &binary_read);
        trans_model.Read(ki.Stream(), binary_read);
        am_nnets[i].Read(ki.Stream(), binary_read);
        nnets[i] = &(am_nnets[i].GetNnet());
      }
    } else {
      for (int32 i = 0; i < 2; i++) {
        nnets[i] = new Nnet;
        ReadKaldiObject(nnet_rxfilenames[i], nnets[i]);
      }
    }

    std::vector<BaseFloat> regularizers;
    SplitStringToFloats(train_config.r, ":", false, &regularizers);

    SequentialNnetMultiviewExampleReader example_reader(
        examples_rspecifier);

    std::vector<std::vector<NnetExample> > buffers;
    buffers.resize(2);
    
    BaseFloat corr = 0.0;
    int32 num_batches = 0, num_examples = 0;
    for (; !example_reader.Done(); example_reader.Next(), 
        num_examples++) {
        const NnetMultiviewExample &eg = example_reader.Value();
        for (int32 i = 0; i < 2; i++)
          buffers[i].push_back(eg.views[i]);
        if (buffers[0].size() == 1000) {
          corr += NnetDccaTrainer::ComputeDccaObjf(nnets, buffers, regularizers, updater_config);
          num_batches++;
          for (int32 i = 0; i < 2; i++) 
            buffers[i].clear();
        }

        if (num_examples % 5000 == 0 && num_examples > 0)
          KALDI_LOG << "Saw " << num_examples << " examples, average "
                    << "correlation is " << corr / num_batches;
    }

    if (!buffers[0].empty()) {
      corr += NnetDccaTrainer::ComputeDccaObjf(nnets, buffers, regularizers, updater_config);
      num_batches++;
    }

    KALDI_LOG << "Saw " << num_examples << " examples, average "
      << "correlation is " << corr / num_batches;

    std::cout << (corr / num_batches) << "\n";
    return (num_examples == 0 ? 1 : 0);
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}

