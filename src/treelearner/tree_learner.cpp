/*!
 * Copyright (c) 2016 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#include <LightGBM/tree_learner.h>

#include <string>

#include "gpu_tree_learner.h"
#ifdef USE_METAL
#include "metal_tree_learner.h"
#endif
#include "linear_tree_learner.h"
#include "parallel_tree_learner.h"
#include "serial_tree_learner.h"
#include "cuda/cuda_single_gpu_tree_learner.hpp"

namespace LightGBM {

TreeLearner* TreeLearner::CreateTreeLearner(const std::string& learner_type, const std::string& device_type,
                                            const Config* config, const bool boosting_on_cuda) {
  if (device_type == std::string("cpu")) {
    if (learner_type == std::string("serial")) {
      if (config->linear_tree) {
        return new LinearTreeLearner<SerialTreeLearner>(config);
      } else {
        return new SerialTreeLearner(config);
      }
    } else if (learner_type == std::string("feature")) {
      return new FeatureParallelTreeLearner<SerialTreeLearner>(config);
    } else if (learner_type == std::string("data")) {
      return new DataParallelTreeLearner<SerialTreeLearner>(config);
    } else if (learner_type == std::string("voting")) {
      return new VotingParallelTreeLearner<SerialTreeLearner>(config);
    }
  } else if (device_type == std::string("gpu")) {
    if (learner_type == std::string("serial")) {
      if (config->linear_tree) {
        return new LinearTreeLearner<GPUTreeLearner>(config);
      } else {
        return new GPUTreeLearner(config);
      }
    } else if (learner_type == std::string("feature")) {
      return new FeatureParallelTreeLearner<GPUTreeLearner>(config);
    } else if (learner_type == std::string("data")) {
      return new DataParallelTreeLearner<GPUTreeLearner>(config);
    } else if (learner_type == std::string("voting")) {
      return new VotingParallelTreeLearner<GPUTreeLearner>(config);
    }
  } else if (device_type == std::string("cuda")) {
    if (learner_type == std::string("serial")) {
      return new CUDASingleGPUTreeLearner(config, boosting_on_cuda);
    } else {
      Log::Fatal("Currently cuda version only supports training on a single machine.");
    }
  } else if (device_type == std::string("metal")) {
#ifdef USE_METAL
    if (learner_type == std::string("serial")) {
      return new MetalTreeLearner(config);
    }
    Log::Fatal("Experimental Metal backend currently supports only the serial tree learner.");
#else
    Log::Fatal("Metal backend was not enabled in this build. Rebuild with USE_METAL=ON.");
#endif
  }
  return nullptr;
}

}  // namespace LightGBM
