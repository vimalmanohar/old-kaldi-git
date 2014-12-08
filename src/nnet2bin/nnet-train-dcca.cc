// nnet2bin/nnet-train-dcca.cc

// Copyright 2012  Johns Hopkins University (author: Daniel Povey)
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
        "Train two neural networks with Deep Canonical Correlation Analysis "
        "(DCCA) objective.\n" 
        "By default reads/writes model file (.mdl) but with --raw=true,\n"
        "reads/writes raw-nnet.\n"
        "\n"
        "Usage:  nnet-train-dcca [options] <model-in-1> <model-in-2> "
        "<training-examples-in-1> <training-examples-in-2> "
        "<model-out-1> <model-out-2>\n"
        "\n"
        "e.g.:\n"
        " nnet-train-dcca 1.view1.nnet 1.view2.nnet ark:egs1.ark ark:egs2.ark "
        "2.view1.nnet 2.view2.nnet\n";
    
    bool binary_write = true;
    int32 srand_seed = 0;
    std::string use_gpu = "yes";
    bool raw = false;    
    NnetDccaTrainerConfig train_config;
    
    ParseOptions po(usage);
    po.Register("raw", &raw,
                "If true, read/write raw neural net rather than .mdl");
    po.Register("binary", &binary_write, "Write output in binary mode");
    po.Register("srand", &srand_seed, "Seed for random number generator "
                "(relevant if you have layers of type AffineComponentPreconditioned "
                "with l2-penalty != 0.0");
    po.Register("use-gpu", &use_gpu, "yes|no|optional, only has effect if compiled with CUDA"); 
 
    train_config.Register(&po);
    
    po.Read(argc, argv);
    
    if (po.NumArgs() != 5) {
      po.PrintUsage();
      exit(1);
    }
    srand(srand_seed);
    
#if HAVE_CUDA==1
    CuDevice::Instantiate().SelectGpuId(use_gpu);
#endif
    
    std::vector<std::string> nnet_rxfilenames;
    std::string examples_rspecifier = po.GetArg(3);
    std::vector<std::string> nnet_wxfilenames;
    nnet_rxfilenames.push_back(po.GetArg(1));
    nnet_rxfilenames.push_back(po.GetArg(2));
    nnet_wxfilenames.push_back(po.GetArg(4));
    nnet_wxfilenames.push_back(po.GetArg(5));
    
    std::vector<AmNnet> am_nnets(2);
    std::vector<Nnet*> nnets(2);

    int64 num_examples = 0;
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

    SequentialNnetMultiviewExampleReader example_reader(
        examples_rspecifier);

    { // want to make sure this object deinitializes before
      // we write the model, as it does something in the destructor.
      NnetDccaTrainer trainer(train_config,
                              nnets);

      for (; !example_reader.Done(); example_reader.Next(), 
          num_examples++) {
        const NnetMultiviewExample &eg = example_reader.Value();
        trainer.TrainOnExample(eg);
      }

      if (!raw) {
        for (int32 n = 0; n < 2; n++) {
          Output ko(nnet_wxfilenames[n], binary_write);
          trans_model.Write(ko.Stream(), binary_write);
          am_nnets[n].Write(ko.Stream(), binary_write);
        }
      } else {
        for (int32 n = 0; n < 2; n++) {
          WriteKaldiObject(*nnets[n], nnet_wxfilenames[n], binary_write);
        }
      }
    }
#if HAVE_CUDA==1
    CuDevice::Instantiate().PrintProfile();
#endif
    
    KALDI_LOG << "Finished training, processed " << num_examples
              << " training examples.";
    return (num_examples == 0 ? 1 : 0);
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}

