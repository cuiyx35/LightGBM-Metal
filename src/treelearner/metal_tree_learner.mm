/*
 * Experimental Apple Metal histogram backend for LightGBM 4.7.0.
 * Copyright (c) 2026 The LightGBM-M experiment contributors.
 * Licensed under the MIT License. See LICENSE in the project root.
 */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <LightGBM/utils/log.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "metal_tree_learner.h"

namespace LightGBM {

namespace {

constexpr uint32_t kBins = 256;
constexpr uint32_t kRowsPerShard = 256;

struct MetalParams {
  uint32_t selected_rows;
  uint32_t features;
  uint32_t rows_per_shard;
  float gradient_scale;
  float hessian_scale;
};

const char* kMetalSource = R"METAL(
#include <metal_stdlib>
using namespace metal;

struct MetalParams {
  uint selected_rows;
  uint features;
  uint rows_per_shard;
  float gradient_scale;
  float hessian_scale;
};

kernel void lightgbm_histogram_fixed32(
    device const uchar* feature_bins [[buffer(0)]],
    device const float* gradients [[buffer(1)]],
    device const float* hessians [[buffer(2)]],
    device const uint* row_indices [[buffer(3)]],
    device const uchar* feature_mask [[buffer(4)]],
    device atomic_int* gradient_hist [[buffer(5)]],
    device atomic_int* hessian_hist [[buffer(6)]],
    constant MetalParams& params [[buffer(7)]],
    uint index [[thread_position_in_grid]]) {
  const uint total = params.selected_rows * params.features;
  if (index >= total) return;
  const uint selected_index = index / params.features;
  const uint feature = index - selected_index * params.features;
  if (!feature_mask[feature]) return;
  const uint row = row_indices[selected_index];
  const uint bin = feature_bins[row * params.features + feature];
  const uint shard = selected_index / params.rows_per_shard;
  const uint offset = (shard * params.features + feature) * 256 + bin;
  atomic_fetch_add_explicit(&gradient_hist[offset], int(rint(gradients[row] * params.gradient_scale)), memory_order_relaxed);
  atomic_fetch_add_explicit(&hessian_hist[offset], int(rint(hessians[row] * params.hessian_scale)), memory_order_relaxed);
}
)METAL";

std::string ErrorDescription(NSError* error) {
  return error ? std::string([[error description] UTF8String]) : "unknown Metal error";
}

}  // namespace

class MetalHistogramEngine {
 public:
  MetalHistogramEngine() {
    device_ = MTLCreateSystemDefaultDevice();
    if (!device_) {
      Log::Fatal("No Apple Metal device is available");
    }
    NSError* error = nil;
    NSString* source = [NSString stringWithUTF8String:kMetalSource];
    id<MTLLibrary> library = [device_ newLibraryWithSource:source options:nil error:&error];
    if (!library) {
      Log::Fatal("Metal shader compilation failed: %s", ErrorDescription(error).c_str());
    }
    id<MTLFunction> function = [library newFunctionWithName:@"lightgbm_histogram_fixed32"];
    if (!function) {
      Log::Fatal("Metal histogram function is missing");
    }
    pipeline_ = [device_ newComputePipelineStateWithFunction:function error:&error];
    if (!pipeline_) {
      Log::Fatal("Metal pipeline creation failed: %s", ErrorDescription(error).c_str());
    }
    queue_ = [device_ newCommandQueue];
    if (!queue_) {
      Log::Fatal("Metal command queue creation failed");
    }
    Log::Info("Experimental Metal device: %s", [[device_ name] UTF8String]);
  }

  bool MirrorDataset(const Dataset* dataset, const std::vector<int>& groups) {
    active_ = false;
    matrix_ = nil;
    gradient_ = nil;
    hessian_ = nil;
    indices_ = nil;
    mask_ = nil;
    output_gradient_ = nil;
    output_hessian_ = nil;
    groups_ = static_cast<uint32_t>(groups.size());
    rows_ = static_cast<uint32_t>(dataset->num_data());
    if (groups_ == 0 || rows_ == 0) return false;
    const uint64_t matrix_elements = uint64_t(rows_) * groups_;
    if (matrix_elements > std::numeric_limits<uint32_t>::max() ||
        matrix_elements > [device_ maxBufferLength]) {
      Log::Warning("Metal matrix exceeds the current 32-bit grid or device buffer limit; using CPU");
      return false;
    }
    const uint64_t shards = (uint64_t(rows_) + kRowsPerShard - 1) / kRowsPerShard;
    const uint64_t output_bytes = shards * groups_ * kBins * sizeof(float);
    if (output_bytes > [device_ maxBufferLength] ||
        shards * groups_ * kBins > std::numeric_limits<uint32_t>::max()) {
      Log::Warning("Metal histogram buffer exceeds device limit; using CPU");
      return false;
    }
    matrix_ = [device_ newBufferWithLength:matrix_elements options:MTLResourceStorageModeShared];
    gradient_ = [device_ newBufferWithLength:uint64_t(rows_) * sizeof(float)
                                   options:MTLResourceStorageModeShared];
    hessian_ = [device_ newBufferWithLength:uint64_t(rows_) * sizeof(float)
                                  options:MTLResourceStorageModeShared];
    indices_ = [device_ newBufferWithLength:uint64_t(rows_) * sizeof(uint32_t)
                                  options:MTLResourceStorageModeShared];
    mask_ = [device_ newBufferWithLength:groups_ options:MTLResourceStorageModeShared];
    output_gradient_ = [device_ newBufferWithLength:output_bytes options:MTLResourceStorageModeShared];
    output_hessian_ = [device_ newBufferWithLength:output_bytes options:MTLResourceStorageModeShared];
    if (!matrix_ || !gradient_ || !hessian_ || !indices_ || !mask_ ||
        !output_gradient_ || !output_hessian_) {
      Log::Warning("Metal buffer allocation failed; using CPU");
      return false;
    }

    std::vector<std::unique_ptr<BinIterator>> iterators;
    iterators.reserve(groups_);
    for (int group : groups) {
      iterators.emplace_back(dataset->FeatureGroupIterator(group));
      if (!iterators.back()) {
        Log::Warning("Metal group iterator unavailable; using CPU");
        return false;
      }
    }
    auto* matrix = static_cast<uint8_t*>([matrix_ contents]);
    for (uint32_t row = 0; row < rows_; ++row) {
      for (uint32_t feature = 0; feature < groups_; ++feature) {
        const uint32_t bin = iterators[feature]->RawGet(row);
        if (bin >= kBins || bin >= static_cast<uint32_t>(dataset->FeatureGroupNumBin(groups[feature]))) {
          Log::Warning("Metal bin is outside its feature group; using CPU");
          return false;
        }
        matrix[uint64_t(row) * groups_ + feature] = static_cast<uint8_t>(bin);
      }
    }
    active_ = true;
    Log::Info("Metal mirrored %u rows and %u dense feature groups", rows_, groups_);
    return true;
  }

  bool active() const { return active_; }

  void SetGradients(const score_t* gradients, const score_t* hessians) {
    if (!active_) return;
    auto* gpu_gradient = static_cast<float*>([gradient_ contents]);
    auto* gpu_hessian = static_cast<float*>([hessian_ contents]);
    double max_abs_gradient = 0.0;
    double max_abs_hessian = 0.0;
    for (uint32_t row = 0; row < rows_; ++row) {
      gpu_gradient[row] = static_cast<float>(gradients[row]);
      gpu_hessian[row] = static_cast<float>(hessians[row]);
      max_abs_gradient = std::max(max_abs_gradient, std::abs(double(gradients[row])));
      max_abs_hessian = std::max(max_abs_hessian, std::abs(double(hessians[row])));
    }
    auto safe_scale = [](double max_abs) -> float {
      if (!std::isfinite(max_abs)) return 0.0f;
      if (max_abs == 0.0) return 100000000.0f;
      const double limit = double(std::numeric_limits<int32_t>::max()) /
                           (double(kRowsPerShard) * max_abs * 1.01);
      return limit >= 1.0 ? static_cast<float>(std::min(100000000.0, std::floor(limit))) : 0.0f;
    };
    gradient_scale_ = safe_scale(max_abs_gradient);
    hessian_scale_ = safe_scale(max_abs_hessian);
  }

  bool Build(const data_size_t* row_indices, data_size_t selected_rows,
             const std::vector<uint8_t>& group_mask,
             const Dataset* dataset, const std::vector<int>& groups,
             hist_t* destination) {
    @autoreleasepool {
    if (!active_ || selected_rows <= 0 || gradient_scale_ == 0.0f || hessian_scale_ == 0.0f) return false;
    if (std::none_of(group_mask.begin(), group_mask.end(), [](uint8_t used) { return used != 0; })) {
      return false;
    }
    const uint64_t work_items = uint64_t(selected_rows) * groups_;
    if (work_items > std::numeric_limits<uint32_t>::max()) return false;
    const uint32_t shards = (uint64_t(selected_rows) + kRowsPerShard - 1) / kRowsPerShard;
    const uint64_t output_elements = uint64_t(shards) * groups_ * kBins;
    const size_t output_bytes = static_cast<size_t>(output_elements * sizeof(float));
    auto* gpu_indices = static_cast<uint32_t*>([indices_ contents]);
    for (data_size_t row = 0; row < selected_rows; ++row) {
      gpu_indices[row] = row_indices ? static_cast<uint32_t>(row_indices[row]) : static_cast<uint32_t>(row);
      if (gpu_indices[row] >= rows_) {
        Log::Fatal("Metal leaf index is outside the mirrored dataset");
      }
    }
    std::memcpy([mask_ contents], group_mask.data(), groups_);
    std::memset([output_gradient_ contents], 0, output_bytes);
    std::memset([output_hessian_ contents], 0, output_bytes);

    const MetalParams params{static_cast<uint32_t>(selected_rows), groups_, kRowsPerShard,
                             gradient_scale_, hessian_scale_};
    id<MTLCommandBuffer> command = [queue_ commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:pipeline_];
    [encoder setBuffer:matrix_ offset:0 atIndex:0];
    [encoder setBuffer:gradient_ offset:0 atIndex:1];
    [encoder setBuffer:hessian_ offset:0 atIndex:2];
    [encoder setBuffer:indices_ offset:0 atIndex:3];
    [encoder setBuffer:mask_ offset:0 atIndex:4];
    [encoder setBuffer:output_gradient_ offset:0 atIndex:5];
    [encoder setBuffer:output_hessian_ offset:0 atIndex:6];
    [encoder setBytes:&params length:sizeof(params) atIndex:7];
    const NSUInteger threads = std::min<NSUInteger>(256, [pipeline_ maxTotalThreadsPerThreadgroup]);
    [encoder dispatchThreads:MTLSizeMake(work_items, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    [encoder endEncoding];
    [command commit];
    [command waitUntilCompleted];
    if ([command status] != MTLCommandBufferStatusCompleted) {
      Log::Fatal("Metal histogram command failed: %s", ErrorDescription([command error]).c_str());
    }
    const auto* gpu_gradient = static_cast<const int32_t*>([output_gradient_ contents]);
    const auto* gpu_hessian = static_cast<const int32_t*>([output_hessian_ contents]);
    for (uint32_t feature = 0; feature < groups_; ++feature) {
      if (!group_mask[feature]) continue;
      const int group = groups[feature];
      hist_t* group_histogram = destination + dataset->GroupBinBoundary(group) * 2;
      const uint32_t group_bins = dataset->FeatureGroupNumBin(group);
      for (uint32_t bin = 0; bin < group_bins; ++bin) {
        double sum_gradient = 0.0;
        double sum_hessian = 0.0;
        for (uint32_t shard = 0; shard < shards; ++shard) {
          const uint64_t offset = (uint64_t(shard) * groups_ + feature) * kBins + bin;
          sum_gradient += double(gpu_gradient[offset]) / gradient_scale_;
          sum_hessian += double(gpu_hessian[offset]) / hessian_scale_;
        }
        GET_GRAD(group_histogram, bin) = sum_gradient;
        GET_HESS(group_histogram, bin) = sum_hessian;
      }
    }
    return true;
    }
  }

 private:
  __strong id<MTLDevice> device_ = nil;
  __strong id<MTLComputePipelineState> pipeline_ = nil;
  __strong id<MTLCommandQueue> queue_ = nil;
  __strong id<MTLBuffer> matrix_ = nil;
  __strong id<MTLBuffer> gradient_ = nil;
  __strong id<MTLBuffer> hessian_ = nil;
  __strong id<MTLBuffer> indices_ = nil;
  __strong id<MTLBuffer> mask_ = nil;
  __strong id<MTLBuffer> output_gradient_ = nil;
  __strong id<MTLBuffer> output_hessian_ = nil;
  uint32_t rows_ = 0;
  uint32_t groups_ = 0;
  bool active_ = false;
  float gradient_scale_ = 0.0f;
  float hessian_scale_ = 0.0f;
};

MetalTreeLearner::MetalTreeLearner(const Config* config)
    : SerialTreeLearner(config), engine_(new MetalHistogramEngine()) {
  const char* force_cpu = std::getenv("LGBM_METAL_FORCE_CPU");
  force_cpu_ = force_cpu != nullptr && std::strcmp(force_cpu, "1") == 0;
  const char* compare_hist = std::getenv("LGBM_METAL_COMPARE_HIST");
  compare_hist_ = compare_hist != nullptr && std::strcmp(compare_hist, "1") == 0;
  if (force_cpu_) {
    Log::Info("Metal diagnostic mode: CPU histograms are forced");
  }
}

MetalTreeLearner::~MetalTreeLearner() = default;

void MetalTreeLearner::Init(const Dataset* train_data, bool is_constant_hessian) {
  SerialTreeLearner::Init(train_data, is_constant_hessian);
  if (!force_cpu_) BuildMirror();
}

void MetalTreeLearner::ResetTrainingDataInner(const Dataset* train_data,
                                               bool is_constant_hessian,
                                               bool reset_multi_val_bin) {
  SerialTreeLearner::ResetTrainingDataInner(train_data, is_constant_hessian, reset_multi_val_bin);
  if (!force_cpu_) BuildMirror();
}

void MetalTreeLearner::BuildMirror() {
  metal_groups_.clear();
  group_to_slot_.assign(train_data_->num_feature_groups(), -1);
  group_feature_count_.assign(train_data_->num_feature_groups(), 0);
  for (int feature = 0; feature < num_features_; ++feature) {
    ++group_feature_count_[train_data_->Feature2Group(feature)];
  }
  for (int group = 0; group < train_data_->num_feature_groups(); ++group) {
    if (group_feature_count_[group] == 1 && !train_data_->IsMultiGroup(group) &&
        train_data_->FeatureGroupNumBin(group) <= static_cast<int>(kBins)) {
      group_to_slot_[group] = static_cast<int>(metal_groups_.size());
      metal_groups_.push_back(group);
    }
  }
  if (!engine_->MirrorDataset(train_data_, metal_groups_)) {
    metal_groups_.clear();
    std::fill(group_to_slot_.begin(), group_to_slot_.end(), -1);
  }
  logged_dispatch_ = false;
  compared_hist_ = false;
}

void MetalTreeLearner::BeforeTrain() {
  SerialTreeLearner::BeforeTrain();
  if (engine_->active() && !share_state_->is_constant_hessian) {
    engine_->SetGradients(gradients_, hessians_);
  }
}

bool MetalTreeLearner::BuildLeafHistogram(const data_size_t* row_indices,
                                           data_size_t row_count,
                                           const std::vector<uint8_t>& group_mask,
                                           hist_t* destination,
                                           double leaf_gradient,
                                           double leaf_hessian) {
  if (share_state_->is_constant_hessian) return false;
  const bool used = engine_->Build(row_indices, row_count, group_mask,
                                   train_data_, metal_groups_, destination);
  if (used) {
    for (int feature = 0; feature < num_features_; ++feature) {
      const int group = train_data_->Feature2Group(feature);
      const int slot = group_to_slot_[group];
      if (slot < 0 || !group_mask[slot]) continue;
      const int frequent_bin = train_data_->FeatureBinMapper(feature)->GetMostFreqBin();
      if (frequent_bin <= 0) continue;
      hist_t* group_histogram = destination + train_data_->GroupBinBoundary(group) * 2;
      const int group_bins = train_data_->FeatureGroupNumBin(group);
      double sum_gradient = 0.0;
      double sum_hessian = 0.0;
      int correction_bin = -1;
      double largest_hessian = -1.0;
      for (int bin = 0; bin < group_bins; ++bin) {
        sum_gradient += GET_GRAD(group_histogram, bin);
        sum_hessian += GET_HESS(group_histogram, bin);
        if (bin > 0 && bin != frequent_bin + 1 &&
            GET_HESS(group_histogram, bin) > largest_hessian) {
          correction_bin = bin;
          largest_hessian = GET_HESS(group_histogram, bin);
        }
      }
      if (correction_bin >= 0) {
        GET_GRAD(group_histogram, correction_bin) += leaf_gradient - sum_gradient;
        GET_HESS(group_histogram, correction_bin) += leaf_hessian - sum_hessian;
      }
    }
  }
  if (used && !logged_dispatch_) {
    Log::Info("Metal histogram kernel dispatched for LightGBM tree training");
    logged_dispatch_ = true;
  }
  return used;
}

void MetalTreeLearner::ConstructHistograms(const std::vector<int8_t>& is_feature_used,
                                           bool use_subtract) {
  if (force_cpu_ || !engine_->active() || share_state_->is_constant_hessian) {
    SerialTreeLearner::ConstructHistograms(is_feature_used, use_subtract);
    return;
  }
  std::vector<uint8_t> group_mask(metal_groups_.size(), 0);
  std::vector<int8_t> cpu_features = is_feature_used;
  for (int feature = 0; feature < num_features_; ++feature) {
    if (!is_feature_used[feature]) continue;
    const int slot = group_to_slot_[train_data_->Feature2Group(feature)];
    if (slot >= 0) {
      group_mask[slot] = 1;
      cpu_features[feature] = 0;
    }
  }
  hist_t* smaller = smaller_leaf_histogram_array_[0].RawData() - kHistOffset;
  const bool smaller_gpu = BuildLeafHistogram(smaller_leaf_splits_->data_indices(),
                                               smaller_leaf_splits_->num_data_in_leaf(),
                                               group_mask, smaller,
                                               smaller_leaf_splits_->sum_gradients(),
                                               smaller_leaf_splits_->sum_hessians());
  if (smaller_gpu && compare_hist_ && !compared_hist_) {
    std::vector<int8_t> gpu_features(num_features_, 0);
    for (int feature = 0; feature < num_features_; ++feature) {
      const int slot = group_to_slot_[train_data_->Feature2Group(feature)];
      if (slot >= 0 && group_mask[slot]) gpu_features[feature] = is_feature_used[feature];
    }
    std::vector<hist_t> cpu_hist(train_data_->NumTotalBin() * 2 + kHistOffset, 0.0);
    train_data_->ConstructHistograms<false, 0>(
        gpu_features, smaller_leaf_splits_->data_indices(),
        smaller_leaf_splits_->num_data_in_leaf(), gradients_, hessians_,
        ordered_gradients_.data(), ordered_hessians_.data(), share_state_.get(),
        cpu_hist.data());
    std::vector<hist_t> metal_hist(cpu_hist.size(), 0.0);
    for (size_t slot = 0; slot < metal_groups_.size(); ++slot) {
      if (!group_mask[slot]) continue;
      const int group = metal_groups_[slot];
      const uint64_t base = train_data_->GroupBinBoundary(group) * 2;
      const uint64_t length = uint64_t(train_data_->FeatureGroupNumBin(group)) * 2;
      std::memcpy(metal_hist.data() + base, smaller + base, length * sizeof(hist_t));
    }
    for (int feature = 0; feature < num_features_; ++feature) {
      if (!gpu_features[feature]) continue;
      const auto offset = smaller_leaf_histogram_array_[feature].RawData() - smaller;
      train_data_->FixHistogram(feature, smaller_leaf_splits_->sum_gradients(),
                                smaller_leaf_splits_->sum_hessians(),
                                cpu_hist.data() + offset);
      train_data_->FixHistogram(feature, smaller_leaf_splits_->sum_gradients(),
                                smaller_leaf_splits_->sum_hessians(),
                                metal_hist.data() + offset);
    }
    double max_gradient_error = 0.0;
    double max_hessian_error = 0.0;
    double max_gradient_scaled_error = 0.0;
    double max_hessian_scaled_error = 0.0;
    int max_group = -1;
    int max_bin = -1;
    int max_hessian_group = -1;
    int max_hessian_bin = -1;
    for (size_t slot = 0; slot < metal_groups_.size(); ++slot) {
      if (!group_mask[slot]) continue;
      const int group = metal_groups_[slot];
      const uint64_t base = train_data_->GroupBinBoundary(group) * 2;
      const hist_t* cpu_group = cpu_hist.data() + base;
      const hist_t* metal_group = metal_hist.data() + base;
      for (int bin = 0; bin < train_data_->FeatureGroupNumBin(group); ++bin) {
        const double cpu_g = GET_GRAD(cpu_group, bin);
        const double cpu_h = GET_HESS(cpu_group, bin);
        const double metal_g = GET_GRAD(metal_group, bin);
        const double metal_h = GET_HESS(metal_group, bin);
        const double ge = std::abs(cpu_g - metal_g);
        const double he = std::abs(cpu_h - metal_h);
        if (ge > max_gradient_error) {
          max_gradient_error = ge;
          max_group = group;
          max_bin = bin;
        }
        if (he > max_hessian_error) {
          max_hessian_error = he;
          max_hessian_group = group;
          max_hessian_bin = bin;
        }
        max_gradient_scaled_error = std::max(max_gradient_scaled_error,
                                              ge / std::max(1.0, std::abs(cpu_g)));
        max_hessian_scaled_error = std::max(max_hessian_scaled_error,
                                             he / std::max(1.0, std::abs(cpu_h)));
      }
    }
    Log::Info("Metal histogram diagnostic after FixHistogram: max_abs_gradient=%.9g max_abs_hessian=%.9g max_scaled_gradient=%.9g max_scaled_hessian=%.9g max_gradient_group=%d max_gradient_bin=%d",
              max_gradient_error, max_hessian_error,
              max_gradient_scaled_error, max_hessian_scaled_error,
              max_group, max_bin);
    if (max_group >= 0) {
      const uint64_t base = train_data_->GroupBinBoundary(max_group) * 2;
      const hist_t* cpu_group = cpu_hist.data() + base;
      const hist_t* metal_group = metal_hist.data() + base;
      Log::Info("Metal diagnostic group=%d bins=%d gradient_bin=%d cpu_gradient=%.9g metal_gradient=%.9g cpu_hessian=%.9g metal_hessian=%.9g max_hessian_group=%d max_hessian_bin=%d",
                max_group, train_data_->FeatureGroupNumBin(max_group), max_bin,
                GET_GRAD(cpu_group, max_bin), GET_GRAD(metal_group, max_bin),
                GET_HESS(cpu_group, max_bin), GET_HESS(metal_group, max_bin),
                max_hessian_group, max_hessian_bin);
      for (int feature = 0; feature < num_features_; ++feature) {
        if (train_data_->Feature2Group(feature) != max_group) continue;
        const int real_feature = train_data_->RealFeatureIndex(feature);
        Log::Info("Metal diagnostic feature=%d name=%s subfeature=%d most_freq_bin=%d feature_bins=%d",
                  feature, train_data_->feature_names()[real_feature].c_str(),
                  train_data_->Feature2SubFeature(feature),
                  train_data_->FeatureBinMapper(feature)->GetMostFreqBin(),
                  train_data_->FeatureNumBin(feature));
      }
    }
    compared_hist_ = true;
  }
  train_data_->ConstructHistograms<false, 0>(
      smaller_gpu ? cpu_features : is_feature_used,
      smaller_leaf_splits_->data_indices(), smaller_leaf_splits_->num_data_in_leaf(),
      gradients_, hessians_, ordered_gradients_.data(), ordered_hessians_.data(),
      share_state_.get(), smaller);

  if (larger_leaf_histogram_array_ != nullptr && !use_subtract) {
    hist_t* larger = larger_leaf_histogram_array_[0].RawData() - kHistOffset;
    const bool larger_gpu = BuildLeafHistogram(larger_leaf_splits_->data_indices(),
                                                larger_leaf_splits_->num_data_in_leaf(),
                                                group_mask, larger,
                                                larger_leaf_splits_->sum_gradients(),
                                                larger_leaf_splits_->sum_hessians());
    train_data_->ConstructHistograms<false, 0>(
        larger_gpu ? cpu_features : is_feature_used,
        larger_leaf_splits_->data_indices(), larger_leaf_splits_->num_data_in_leaf(),
        gradients_, hessians_, ordered_gradients_.data(), ordered_hessians_.data(),
        share_state_.get(), larger);
  }
}

}  // namespace LightGBM
