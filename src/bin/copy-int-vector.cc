// bin/copy-int-vector.cc

// Copyright 2009-2011  Microsoft Corporation

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
#include "matrix/kaldi-vector.h"
#include "transform/transform-common.h"

namespace kaldi {
  typedef unordered_map<std::string, std::string, StringHasher> string_map;
  typedef unordered_set<std::string, StringHasher> string_set;

  int32 CopySubsetVectors(std::string filename, 
      SequentialInt32VectorReader &reader,
      Int32VectorWriter &writer,
      bool include = true, bool ignore_missing = false) {
    string_set subset;
    
    if (filename != "") { 
      bool binary;
      Input ki(filename, &binary);
      KALDI_ASSERT(!binary);
      std::string line;
      while (std::getline(ki.Stream(), line)) {
        std::vector<std::string> split_line;
        SplitStringToVector(line, " \t\r", true, &split_line);
        if(split_line.empty()) {
          KALDI_ERR << "Unable to parse line \"" << line << "\" encountered in input in " << filename;
        }
        subset.insert(split_line[0]);
      }
    }

    int32 num_total = 0, num_success = 0;
    for (; !reader.Done(); reader.Next(), num_total++) {
      std::string key = reader.Key();
      if (include && subset.count(key) > 0) {
        writer.Write(key, reader.Value());
        num_success++;
      } else if (!include && subset.count(key) == 0) {
        writer.Write(key, reader.Value());
        num_success++;
      }
    }

    KALDI_LOG << "Copied " << num_success << " out of " << num_total 
              << " int32 vectors.";

    if (ignore_missing) return 0;

    return (num_success != 0 ? 0 : 1);
  }
}

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;

    const char *usage =
        "Copy vectors of integers, or archives of vectors of integers \n"
        "(e.g. alignments)\n"
        "\n"
        "Usage: copy-int-vector [options] (vector-in-rspecifier|vector-in-rxfilename) (vector-out-wspecifier|vector-out-wxfilename)\n"
        " e.g.: copy-int-vector --binary=false foo -\n"
        "   copy-int-vector ark:1.ali ark,t:-\n";
    
    bool binary = true, ignore_missing = false;
    std::string include_rxfilename;
    std::string exclude_rxfilename;
    
    ParseOptions po(usage);

    po.Register("binary", &binary, "Write in binary mode (only relevant if output is a wxfilename)");
    po.Register("include", &include_rxfilename, 
                        "Text file, the first field of each "
                        "line being interpreted as an "
                        "utterance-id whose features will be included");
    po.Register("exclude", &exclude_rxfilename, 
                        "Text file, the first field of each "
                        "line being interpreted as an utterance-id"
                        " whose features will be excluded");
    po.Register("ignore-missing", &ignore_missing,
                        "Exit with status 0 even if no vectors are copied");

    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }
    
    std::string vector_in_fn = po.GetArg(1),
        vector_out_fn = po.GetArg(2);

    // all these "fn"'s are either rspecifiers or filenames.

    bool in_is_rspecifier =
        (ClassifyRspecifier(vector_in_fn, NULL, NULL)
         != kNoRspecifier),
        out_is_wspecifier =
        (ClassifyWspecifier(vector_out_fn, NULL, NULL, NULL)
         != kNoWspecifier);

    if (in_is_rspecifier != out_is_wspecifier)
      KALDI_ERR << "Cannot mix archives with regular files (copying vectors)";
    
    if (!in_is_rspecifier) {
      std::vector<int32> vec;
      {
        bool binary_in;
        Input ki(vector_in_fn, &binary_in);
        ReadIntegerVector(ki.Stream(), binary_in, &vec);
      }
      Output ko(vector_out_fn, binary);
      WriteIntegerVector(ko.Stream(), binary, vec);
      KALDI_LOG << "Copied vector to " << vector_out_fn;
      return 0;
    } else {
      Int32VectorWriter writer(vector_out_fn);
      SequentialInt32VectorReader reader(vector_in_fn);
      
      if (include_rxfilename != "") {
        if (exclude_rxfilename != "") {
          KALDI_ERR << "should not have both --exclude and --include option!";
        }
        return CopySubsetVectors(include_rxfilename, reader, writer, \
          true , ignore_missing);
      } else {
        return CopySubsetVectors(include_rxfilename, reader, writer, \
          true , ignore_missing);
      }
    }
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

