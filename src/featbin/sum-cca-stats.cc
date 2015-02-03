// featbin/sum-cca.cc

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

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "matrix/kaldi-matrix.h"
#include "matrix/kaldi-vector.h"
#include "feat/cca.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;

    const char *usage =
        "Computes the correlation from stats "
        "using the CCA objective. \n"
        "The target dim can be set using --target-dim.\n"
        "\n"
        "Usage: sum-cca <stats-1> <stats-2> ... <stats-N> <stats>\n"
        "e.g. : sum-cca stats-1 stats-2 stats\n";

    ParseOptions po(usage);

    bool binary = true;
    po.Register("binary", &binary, "Write in binary mode (only relevant if output is a wxfilename)");
  
    po.Read(argc, argv);

    if (po.NumArgs() < 2) {
      po.PrintUsage();
      exit(1);
    }
    
    CcaStats stats;

    {
      bool binary;
      Input ki(po.GetArg(1), &binary);
      stats.Read(ki.Stream(), binary);
    }

    for (int32 i = 2; i < po.NumArgs(); i++) {
      CcaStats stats_i;
      bool binary;
      Input ki(po.GetArg(i), &binary);
      stats_i.Read(ki.Stream(), binary);
      stats.Sum(stats_i);
    }

    Output ko(po.GetArg(po.NumArgs()), binary);
    stats.Write(ko.Stream(), binary);

    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}



