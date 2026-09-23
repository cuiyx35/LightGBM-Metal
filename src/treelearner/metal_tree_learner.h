/*
 * Experimental Apple Metal histogram backend for LightGBM 4.7.0.
 * Copyright (c) 2026 The LightGBM-M experiment contributors.
 * Licensed under the MIT License. See LICENSE in the project root.
 */
#ifndef LIGHTGBM_SRC_TREELEARNER_METAL_TREE_LEARNER_H_
#define LIGHTGBM_SRC_TREELEARNER_METAL_TREE_LEARNER_H_

#ifdef USE_METAL

#include <memory>
#include <vector>

#include "serial_tree_learner.h"

namespace LightGBM {

class MetalHistogramEngine;

class MetalTreeLearner final : public SerialTreeLearner {
 public:
  explicit MetalTreeLearner(const Config* config);
  ~MetalTreeLearner() override;

  void Init(const Dataset* train_data, bool is_constant_hessian) override;
  void ResetTrainingDataInner(const Dataset* train_data,
                              bool is_constant_hessian,
                              bool reset_multi_val_bin) override;

 protected:
  void BeforeTrain() override;
  void ConstructHistograms(const std::vector<int8_t>& is_feature_used,
                           bool use_subtract) override;

 private:
  void BuildMirror();
  bool BuildLeafHistogram(const data_size_t* row_indices, data_size_t row_count,
                          const std::vector<uint8_t>& group_mask, hist_t* destination,
                          double leaf_gradient, double leaf_hessian);

  std::unique_ptr<MetalHistogramEngine> engine_;
  std::vector<int> metal_groups_;
  std::vector<int> group_to_slot_;
  std::vector<int> group_feature_count_;
  bool logged_dispatch_ = false;
  bool force_cpu_ = false;
  bool compare_hist_ = false;
  bool compared_hist_ = false;
  data_size_t min_leaf_rows_ = 0;
};

}  // namespace LightGBM

#endif  // USE_METAL
#endif  // LIGHTGBM_SRC_TREELEARNER_METAL_TREE_LEARNER_H_
