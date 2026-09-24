/*
 * Experimental Apple Metal histogram backend for LightGBM 4.7.0.
 * Copyright (c) 2026 The LightGBM-M experiment contributors.
 * Licensed under the MIT License. See LICENSE in the project root.
 */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <LightGBM/utils/log.h>

#include <algorithm>
#include <chrono>
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
constexpr uint32_t kDefaultRowsPerShard = 16384;
constexpr uint32_t kDefaultRowsPerChunk = 256;

struct MetalParams {
  uint32_t selected_rows;
  uint32_t features;
  uint32_t matrix_rows;
  uint32_t rows_per_shard;
  uint32_t rows_per_chunk;
  float gradient_scale;
  float hessian_scale;
};

const char* kMetalSource = R"METAL(
#include <metal_stdlib>
using namespace metal;

struct MetalParams {
  uint selected_rows;
  uint features;
  uint matrix_rows;
  uint rows_per_shard;
  uint rows_per_chunk;
  float gradient_scale;
  float hessian_scale;
};

kernel void lightgbm_histogram_grouped(
    device const uchar* feature_bins [[buffer(0)]],
    device const float* gradients [[buffer(1)]],
    device const float* hessians [[buffer(2)]],
    device const uint* row_indices [[buffer(3)]],
    device const uchar* feature_mask [[buffer(4)]],
    device long* gradient_hist [[buffer(5)]],
    device long* hessian_hist [[buffer(6)]],
    constant MetalParams& params [[buffer(7)]],
    uint3 group_position [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
  const uint feature = group_position.x;
  const uint shard = group_position.y;
  if (!feature_mask[feature]) return;
  threadgroup atomic_int chunk_gradient[256];
  threadgroup atomic_int chunk_hessian[256];
  long gradient_total = 0;
  long hessian_total = 0;
  for (uint chunk = 0; chunk < params.rows_per_shard / params.rows_per_chunk; ++chunk) {
    atomic_store_explicit(&chunk_gradient[thread_index], 0, memory_order_relaxed);
    atomic_store_explicit(&chunk_hessian[thread_index], 0, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint selected_index = shard * params.rows_per_shard +
                                chunk * params.rows_per_chunk + thread_index;
    if (thread_index < params.rows_per_chunk && selected_index < params.selected_rows) {
      const uint row = row_indices[selected_index];
      const uint bin = feature_bins[feature * params.matrix_rows + row];
      atomic_fetch_add_explicit(&chunk_gradient[bin],
          int(rint(gradients[row] * params.gradient_scale)), memory_order_relaxed);
      atomic_fetch_add_explicit(&chunk_hessian[bin],
          int(rint(hessians[row] * params.hessian_scale)), memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    gradient_total += atomic_load_explicit(&chunk_gradient[thread_index], memory_order_relaxed);
    hessian_total += atomic_load_explicit(&chunk_hessian[thread_index], memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  const uint output_index = (shard * params.features + feature) * 256 + thread_index;
  gradient_hist[output_index] = gradient_total;
  hessian_hist[output_index] = hessian_total;
}
)METAL";

std::string ErrorDescription(NSError* error) {
  return error ? std::string([[error description] UTF8String]) : "unknown Metal error";
}

using ProfileClock = std::chrono::steady_clock;

double SecondsSince(ProfileClock::time_point start) {
  return std::chrono::duration<double>(ProfileClock::now() - start).count();
}

}  // namespace

class MetalHistogramEngine {
 public:
  MetalHistogramEngine() {
    const char* profile = std::getenv("LGBM_METAL_PROFILE");
    profile_enabled_ = profile != nullptr && std::strcmp(profile, "1") == 0;
    const char* rows_per_chunk = std::getenv("LGBM_METAL_ROWS_PER_CHUNK");
    if (rows_per_chunk != nullptr) {
      char* end = nullptr;
      const long parsed = std::strtol(rows_per_chunk, &end, 10);
      if (*rows_per_chunk == '\0' || *end != '\0' ||
          (parsed != 32 && parsed != 64 && parsed != 128 && parsed != 256)) {
        Log::Fatal("LGBM_METAL_ROWS_PER_CHUNK must be 32, 64, 128, or 256");
      }
      rows_per_chunk_ = static_cast<uint32_t>(parsed);
    }
    const char* rows_per_shard = std::getenv("LGBM_METAL_ROWS_PER_SHARD");
    if (rows_per_shard != nullptr) {
      char* end = nullptr;
      const long parsed = std::strtol(rows_per_shard, &end, 10);
      if (*rows_per_shard == '\0' || *end != '\0' ||
          (parsed != 4096 && parsed != 8192 && parsed != 16384 &&
           parsed != 32768 && parsed != 65536)) {
        Log::Fatal("LGBM_METAL_ROWS_PER_SHARD must be 4096, 8192, 16384, 32768, or 65536");
      }
      rows_per_shard_ = static_cast<uint32_t>(parsed);
    }
    const char* verify_group = std::getenv("LGBM_METAL_VERIFY_QUANTIZED_GROUP");
    if (verify_group != nullptr) {
      if (std::strcmp(verify_group, "all") == 0) {
        verify_quantized_group_ = -2;
      } else {
        char* end = nullptr;
        const long parsed = std::strtol(verify_group, &end, 10);
        if (*verify_group == '\0' || *end != '\0' || parsed < 0 ||
            parsed > std::numeric_limits<int>::max()) {
          Log::Fatal("LGBM_METAL_VERIFY_QUANTIZED_GROUP must be all or a non-negative group id");
        }
        verify_quantized_group_ = static_cast<int>(parsed);
      }
    }
    const char* verify_dispatch = std::getenv("LGBM_METAL_VERIFY_QUANTIZED_DISPATCH");
    if (verify_dispatch != nullptr) {
      char* end = nullptr;
      const long parsed = std::strtol(verify_dispatch, &end, 10);
      if (verify_group == nullptr || *verify_dispatch == '\0' || *end != '\0' ||
          parsed < 1 || parsed > 1000000) {
        Log::Fatal("LGBM_METAL_VERIFY_QUANTIZED_DISPATCH requires a group check and an index from 1 to 1000000");
      }
      verify_quantized_dispatch_ = static_cast<uint64_t>(parsed);
    }
    const auto setup_start = ProfileClock::now();
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
    id<MTLFunction> function = [library newFunctionWithName:@"lightgbm_histogram_grouped"];
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
    setup_seconds_ = SecondsSince(setup_start);
    Log::Info("Experimental Metal device: %s", [[device_ name] UTF8String]);
  }

  ~MetalHistogramEngine() {
    if (inflight_command_) [inflight_command_ waitUntilCompleted];
    if (profile_enabled_) {
      Log::Info("Metal profile: setup=%.6fs mirror=%.6fs gradients=%.6fs preparation=%.6fs gpu_inflight=%.6fs gpu_wait=%.6fs merge=%.6fs dispatches=%llu rows_per_shard=%u allocated=%.1fMiB",
                setup_seconds_, mirror_seconds_, gradient_seconds_, preparation_seconds_,
                gpu_inflight_seconds_, gpu_wait_seconds_, merge_seconds_,
                static_cast<unsigned long long>(dispatches_), rows_per_shard_,
                double(buffer_bytes_) / (1024.0 * 1024.0));
    }
  }

  bool MirrorDataset(const Dataset* dataset, const std::vector<int>& groups) {
    const auto mirror_start = ProfileClock::now();
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
    const uint64_t shards = (uint64_t(rows_) + rows_per_shard_ - 1) / rows_per_shard_;
    const uint64_t output_bytes = shards * groups_ * kBins * sizeof(int64_t);
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
    buffer_bytes_ = matrix_elements + uint64_t(rows_) * (sizeof(float) * 2 + sizeof(uint32_t)) +
                    groups_ + output_bytes * 2;

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
    for (uint32_t feature = 0; feature < groups_; ++feature) {
      for (uint32_t row = 0; row < rows_; ++row) {
        const uint32_t bin = iterators[feature]->RawGet(row);
        if (bin >= kBins || bin >= static_cast<uint32_t>(dataset->FeatureGroupNumBin(groups[feature]))) {
          Log::Warning("Metal bin is outside its feature group; using CPU");
          return false;
        }
        matrix[uint64_t(feature) * rows_ + row] = static_cast<uint8_t>(bin);
      }
    }
    active_ = true;
    if (profile_enabled_) mirror_seconds_ += SecondsSince(mirror_start);
    Log::Info("Metal mirrored %u rows and %u dense feature groups", rows_, groups_);
    return true;
  }

  bool active() const { return active_; }

  void SetGradients(const score_t* gradients, const score_t* hessians) {
    if (!active_) return;
    const auto gradient_start = ProfileClock::now();
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
    auto safe_scale = [this](double max_abs) -> float {
      if (!std::isfinite(max_abs)) return 0.0f;
      if (max_abs == 0.0) return 1.0f;
      const double limit = double(std::numeric_limits<int32_t>::max()) /
                           (double(rows_per_chunk_) * max_abs * 1.01);
      const double scale = std::min(1.0e30, std::floor(limit));
      if (scale < 1.0 || max_abs * scale < 1024.0) return 0.0f;
      return static_cast<float>(scale);
    };
    gradient_scale_ = safe_scale(max_abs_gradient);
    hessian_scale_ = safe_scale(max_abs_hessian);
    if (profile_enabled_) gradient_seconds_ += SecondsSince(gradient_start);
  }

  bool BeginBuild(const data_size_t* row_indices, data_size_t selected_rows,
                  const std::vector<uint8_t>& group_mask) {
    @autoreleasepool {
    if (!active_ || selected_rows <= 0 || gradient_scale_ == 0.0f || hessian_scale_ == 0.0f) return false;
    if (std::none_of(group_mask.begin(), group_mask.end(), [](uint8_t used) { return used != 0; })) {
      return false;
    }
    if (inflight_command_) Log::Fatal("Metal histogram build is already in flight");
    const auto preparation_start = ProfileClock::now();
    inflight_shards_ = (uint64_t(selected_rows) + rows_per_shard_ - 1) / rows_per_shard_;
    inflight_selected_rows_ = static_cast<uint32_t>(selected_rows);
    auto* gpu_indices = static_cast<uint32_t*>([indices_ contents]);
    for (data_size_t row = 0; row < selected_rows; ++row) {
      gpu_indices[row] = row_indices ? static_cast<uint32_t>(row_indices[row]) : static_cast<uint32_t>(row);
      if (gpu_indices[row] >= rows_) {
        Log::Fatal("Metal leaf index is outside the mirrored dataset");
      }
    }
    std::memcpy([mask_ contents], group_mask.data(), groups_);
    const MetalParams params{static_cast<uint32_t>(selected_rows), groups_, rows_,
                             rows_per_shard_, rows_per_chunk_,
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
    if ([pipeline_ maxTotalThreadsPerThreadgroup] < kBins) {
      Log::Fatal("Metal device cannot run the required 256-thread histogram group");
    }
    [encoder dispatchThreadgroups:MTLSizeMake(groups_, inflight_shards_, 1)
        threadsPerThreadgroup:MTLSizeMake(kBins, 1, 1)];
    [encoder endEncoding];
    if (profile_enabled_) preparation_seconds_ += SecondsSince(preparation_start);
    [command commit];
    inflight_command_ = command;
    if (profile_enabled_) inflight_start_ = ProfileClock::now();
    return true;
    }
  }

  void FinishBuild(const std::vector<uint8_t>& group_mask,
                   const Dataset* dataset, const std::vector<int>& groups,
                   hist_t* destination) {
    @autoreleasepool {
    if (!inflight_command_) Log::Fatal("No Metal histogram build is in flight");
    const auto wait_start = ProfileClock::now();
    [inflight_command_ waitUntilCompleted];
    if (profile_enabled_) {
      gpu_wait_seconds_ += SecondsSince(wait_start);
      gpu_inflight_seconds_ += SecondsSince(inflight_start_);
    }
    if ([inflight_command_ status] != MTLCommandBufferStatusCompleted) {
      Log::Fatal("Metal histogram command failed: %s",
                 ErrorDescription([inflight_command_ error]).c_str());
    }
    inflight_command_ = nil;
    const auto* gpu_gradient = static_cast<const int64_t*>([output_gradient_ contents]);
    const auto* gpu_hessian = static_cast<const int64_t*>([output_hessian_ contents]);
    ++completed_builds_;
    if (verify_quantized_group_ != -1 && !verified_quantized_group_ &&
        completed_builds_ == verify_quantized_dispatch_) {
      uint32_t verified_groups = 0;
      uint64_t max_gradient_units_all = 0;
      uint64_t max_hessian_units_all = 0;
      uint32_t mismatched_gradient_bins_all = 0;
      uint32_t mismatched_hessian_bins_all = 0;
      for (uint32_t slot = 0; slot < groups_; ++slot) {
        if (!group_mask[slot] ||
            (verify_quantized_group_ >= 0 && groups[slot] != verify_quantized_group_)) continue;
        std::vector<int64_t> expected_gradient(kBins, 0);
        std::vector<int64_t> expected_hessian(kBins, 0);
        const auto* matrix = static_cast<const uint8_t*>([matrix_ contents]);
        const auto* gradients = static_cast<const float*>([gradient_ contents]);
        const auto* hessians = static_cast<const float*>([hessian_ contents]);
        const auto* indices = static_cast<const uint32_t*>([indices_ contents]);
        for (uint32_t selected = 0; selected < inflight_selected_rows_; ++selected) {
          const uint32_t row = indices[selected];
          const uint32_t bin = matrix[uint64_t(slot) * rows_ + row];
          expected_gradient[bin] += static_cast<int32_t>(
              std::nearbyint(gradients[row] * gradient_scale_));
          expected_hessian[bin] += static_cast<int32_t>(
              std::nearbyint(hessians[row] * hessian_scale_));
        }
        uint64_t max_gradient_units = 0;
        uint64_t max_hessian_units = 0;
        uint32_t mismatched_gradient_bins = 0;
        uint32_t mismatched_hessian_bins = 0;
        for (uint32_t bin = 0; bin < kBins; ++bin) {
          int64_t observed_gradient = 0;
          int64_t observed_hessian = 0;
          for (uint32_t shard = 0; shard < inflight_shards_; ++shard) {
            const uint64_t offset = (uint64_t(shard) * groups_ + slot) * kBins + bin;
            observed_gradient += gpu_gradient[offset];
            observed_hessian += gpu_hessian[offset];
          }
          const uint64_t gradient_units = static_cast<uint64_t>(
              std::llabs(observed_gradient - expected_gradient[bin]));
          const uint64_t hessian_units = static_cast<uint64_t>(
              std::llabs(observed_hessian - expected_hessian[bin]));
          max_gradient_units = std::max(max_gradient_units, gradient_units);
          max_hessian_units = std::max(max_hessian_units, hessian_units);
          mismatched_gradient_bins += gradient_units != 0;
          mismatched_hessian_bins += hessian_units != 0;
        }
        ++verified_groups;
        max_gradient_units_all = std::max(max_gradient_units_all, max_gradient_units);
        max_hessian_units_all = std::max(max_hessian_units_all, max_hessian_units);
        mismatched_gradient_bins_all += mismatched_gradient_bins;
        mismatched_hessian_bins_all += mismatched_hessian_bins;
        if (verify_quantized_group_ >= 0) break;
      }
      if (verified_groups > 0) {
        Log::Info("Metal quantized verification dispatch=%llu groups=%u rows=%u gradient_scale=%.9g hessian_scale=%.9g max_gradient_integer_difference=%llu max_hessian_integer_difference=%llu mismatched_gradient_bins=%u mismatched_hessian_bins=%u",
                  static_cast<unsigned long long>(completed_builds_), verified_groups,
                  inflight_selected_rows_, gradient_scale_,
                  hessian_scale_, static_cast<unsigned long long>(max_gradient_units_all),
                  static_cast<unsigned long long>(max_hessian_units_all),
                  mismatched_gradient_bins_all, mismatched_hessian_bins_all);
        verified_quantized_group_ = true;
      }
    }
    const auto merge_start = ProfileClock::now();
    for (uint32_t feature = 0; feature < groups_; ++feature) {
      if (!group_mask[feature]) continue;
      const int group = groups[feature];
      hist_t* group_histogram = destination + dataset->GroupBinBoundary(group) * 2;
      const uint32_t group_bins = dataset->FeatureGroupNumBin(group);
      for (uint32_t bin = 0; bin < group_bins; ++bin) {
        double sum_gradient = 0.0;
        double sum_hessian = 0.0;
        for (uint32_t shard = 0; shard < inflight_shards_; ++shard) {
          const uint64_t offset = (uint64_t(shard) * groups_ + feature) * kBins + bin;
          sum_gradient += double(gpu_gradient[offset]) / gradient_scale_;
          sum_hessian += double(gpu_hessian[offset]) / hessian_scale_;
        }
        GET_GRAD(group_histogram, bin) = sum_gradient;
        GET_HESS(group_histogram, bin) = sum_hessian;
      }
    }
    if (profile_enabled_) {
      merge_seconds_ += SecondsSince(merge_start);
      ++dispatches_;
    }
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
  __strong id<MTLCommandBuffer> inflight_command_ = nil;
  uint32_t inflight_shards_ = 0;
  uint32_t inflight_selected_rows_ = 0;
  ProfileClock::time_point inflight_start_{};
  uint32_t rows_ = 0;
  uint32_t groups_ = 0;
  bool active_ = false;
  float gradient_scale_ = 0.0f;
  float hessian_scale_ = 0.0f;
  uint32_t rows_per_chunk_ = kDefaultRowsPerChunk;
  uint32_t rows_per_shard_ = kDefaultRowsPerShard;
  int verify_quantized_group_ = -1;
  uint64_t verify_quantized_dispatch_ = 1;
  uint64_t completed_builds_ = 0;
  bool verified_quantized_group_ = false;
  bool profile_enabled_ = false;
  uint64_t buffer_bytes_ = 0;
  uint64_t dispatches_ = 0;
  double setup_seconds_ = 0.0;
  double mirror_seconds_ = 0.0;
  double gradient_seconds_ = 0.0;
  double preparation_seconds_ = 0.0;
  double gpu_inflight_seconds_ = 0.0;
  double gpu_wait_seconds_ = 0.0;
  double merge_seconds_ = 0.0;
};

MetalTreeLearner::MetalTreeLearner(const Config* config)
    : SerialTreeLearner(config), engine_(new MetalHistogramEngine()) {
  const char* force_cpu = std::getenv("LGBM_METAL_FORCE_CPU");
  force_cpu_ = force_cpu != nullptr && std::strcmp(force_cpu, "1") == 0;
  const char* disable_overlap = std::getenv("LGBM_METAL_DISABLE_OVERLAP");
  disable_overlap_ = disable_overlap != nullptr && std::strcmp(disable_overlap, "1") == 0;
  const char* compare_hist = std::getenv("LGBM_METAL_COMPARE_HIST");
  compare_hist_ = compare_hist != nullptr && std::strcmp(compare_hist, "1") == 0;
  const char* min_leaf_rows = std::getenv("LGBM_METAL_MIN_LEAF_ROWS");
  if (min_leaf_rows != nullptr) {
    char* end = nullptr;
    const long parsed = std::strtol(min_leaf_rows, &end, 10);
    if (end == min_leaf_rows || *end != '\0' || parsed < 0 ||
        parsed > std::numeric_limits<data_size_t>::max()) {
      Log::Fatal("LGBM_METAL_MIN_LEAF_ROWS must be a non-negative integer");
    }
    min_leaf_rows_ = static_cast<data_size_t>(parsed);
  }
  if (force_cpu_) {
    Log::Info("Metal diagnostic mode: CPU histograms are forced");
  } else if (disable_overlap_) {
    Log::Info("Metal diagnostic mode: CPU and GPU histogram work is serialized");
  }
}

MetalTreeLearner::~MetalTreeLearner() = default;

void MetalTreeLearner::Init(const Dataset* train_data, bool is_constant_hessian) {
  SerialTreeLearner::Init(train_data, is_constant_hessian);
  if (!force_cpu_ && !is_constant_hessian) BuildMirror();
}

void MetalTreeLearner::ResetTrainingDataInner(const Dataset* train_data,
                                               bool is_constant_hessian,
                                               bool reset_multi_val_bin) {
  SerialTreeLearner::ResetTrainingDataInner(train_data, is_constant_hessian, reset_multi_val_bin);
  if (!force_cpu_ && !is_constant_hessian) BuildMirror();
}

void MetalTreeLearner::BuildMirror() {
  metal_groups_.clear();
  group_to_slot_.assign(train_data_->num_feature_groups(), -1);
  group_feature_count_.assign(train_data_->num_feature_groups(), 0);
  if (!share_state_->is_col_wise) {
    engine_->MirrorDataset(train_data_, metal_groups_);
    Log::Warning("Metal histograms require column-wise training; using CPU. Set force_col_wise=true to enable Metal");
    return;
  }
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
  if (engine_->active() && share_state_->is_col_wise &&
      !share_state_->is_constant_hessian) {
    engine_->SetGradients(gradients_, hessians_);
  }
}

bool MetalTreeLearner::BeginLeafHistogram(const data_size_t* row_indices,
                                          data_size_t row_count,
                                          const std::vector<uint8_t>& group_mask) {
  if (share_state_->is_constant_hessian || row_count < min_leaf_rows_) return false;
  return engine_->BeginBuild(row_indices, row_count, group_mask);
}

void MetalTreeLearner::FinishLeafHistogram(const std::vector<uint8_t>& group_mask,
                                           hist_t* destination,
                                           double leaf_gradient,
                                           double leaf_hessian) {
  engine_->FinishBuild(group_mask, train_data_, metal_groups_, destination);
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
  if (!logged_dispatch_) {
    Log::Info("Metal histogram kernel dispatched for LightGBM tree training");
    logged_dispatch_ = true;
  }
}

void MetalTreeLearner::ConstructHistograms(const std::vector<int8_t>& is_feature_used,
                                           bool use_subtract) {
  if (force_cpu_ || !engine_->active() || !share_state_->is_col_wise ||
      share_state_->is_constant_hessian) {
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
  const bool smaller_gpu = BeginLeafHistogram(smaller_leaf_splits_->data_indices(),
                                              smaller_leaf_splits_->num_data_in_leaf(),
                                              group_mask);
  if (smaller_gpu && disable_overlap_) {
    FinishLeafHistogram(group_mask, smaller, smaller_leaf_splits_->sum_gradients(),
                        smaller_leaf_splits_->sum_hessians());
  }
  train_data_->ConstructHistograms<false, 0>(
      smaller_gpu ? cpu_features : is_feature_used,
      smaller_leaf_splits_->data_indices(), smaller_leaf_splits_->num_data_in_leaf(),
      gradients_, hessians_, ordered_gradients_.data(), ordered_hessians_.data(),
      share_state_.get(), smaller);
  if (smaller_gpu && !disable_overlap_) {
    FinishLeafHistogram(group_mask, smaller, smaller_leaf_splits_->sum_gradients(),
                        smaller_leaf_splits_->sum_hessians());
  }
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
  if (larger_leaf_histogram_array_ != nullptr && !use_subtract) {
    hist_t* larger = larger_leaf_histogram_array_[0].RawData() - kHistOffset;
    const bool larger_gpu = BeginLeafHistogram(larger_leaf_splits_->data_indices(),
                                               larger_leaf_splits_->num_data_in_leaf(),
                                               group_mask);
    if (larger_gpu && disable_overlap_) {
      FinishLeafHistogram(group_mask, larger, larger_leaf_splits_->sum_gradients(),
                          larger_leaf_splits_->sum_hessians());
    }
    train_data_->ConstructHistograms<false, 0>(
        larger_gpu ? cpu_features : is_feature_used,
        larger_leaf_splits_->data_indices(), larger_leaf_splits_->num_data_in_leaf(),
        gradients_, hessians_, ordered_gradients_.data(), ordered_hessians_.data(),
        share_state_.get(), larger);
    if (larger_gpu && !disable_overlap_) {
      FinishLeafHistogram(group_mask, larger, larger_leaf_splits_->sum_gradients(),
                          larger_leaf_splits_->sum_hessians());
    }
  }
}

}  // namespace LightGBM
