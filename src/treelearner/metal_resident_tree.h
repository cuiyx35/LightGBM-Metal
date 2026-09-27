/*
 * Copyright (c) 2026 The LightGBM-M experiment contributors.
 * Licensed under the MIT License. See LICENSE in the project root.
 */
#ifndef LIGHTGBM_SRC_TREELEARNER_METAL_RESIDENT_TREE_H_
#define LIGHTGBM_SRC_TREELEARNER_METAL_RESIDENT_TREE_H_
#ifdef USE_METAL

#include <LightGBM/config.h>
#include <LightGBM/dataset.h>
#include <LightGBM/tree.h>

#include <memory>
#include <string>
#include <vector>

#include "data_partition.hpp"

namespace LightGBM {

// Optional whole-tree Metal engine. A supported tree uses one command buffer;
// the host materializes a regular LightGBM Tree after the device finishes.
class MetalResidentTreeEngine {
 public:
  explicit MetalResidentTreeEngine(const Dataset* data);
  ~MetalResidentTreeEngine();
  std::string UnsupportedReason(const Config& config) const;
  Tree* Train(const Config& config, const score_t* gradients,
              const score_t* hessians, const std::vector<int8_t>& feature_mask,
              DataPartition* partition);

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace LightGBM
#endif
#endif  // LIGHTGBM_SRC_TREELEARNER_METAL_RESIDENT_TREE_H_
