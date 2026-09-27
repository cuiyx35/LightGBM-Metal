/*
 * Copyright (c) 2026 The LightGBM-M experiment contributors.
 * Licensed under the MIT License. See LICENSE in the project root.
 */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "metal_resident_tree.h"
#include "feature_histogram.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

namespace LightGBM {
namespace {

struct ResidentParams {
  uint32_t rows, features, leaves, shards, selected_rows, max_depth;
  uint32_t min_data, max_cat_onehot, max_cat_threshold, min_data_group;
  float inv_g, inv_h, l1, l2, min_hessian, min_gain, max_delta, smooth, cat_l2, cat_smooth;
};
struct ResidentFeature { uint32_t bins, missing, default_bin, categorical, real_feature; };
struct ResidentLeaf {
  uint32_t begin, count, depth, reserved;
  float gradient, hessian, output, pad;
};
struct ResidentCandidate {
  float gain, left_output, right_output, left_hessian, right_hessian;
  uint32_t feature, threshold, default_left, category_bits[8];
};
struct ResidentTrace {
  ResidentCandidate candidate;
  uint32_t leaf, left_count, right_count, valid;
};
struct ResidentState {
  uint32_t selected, right, small, stopped, step, left_count, right_count, root;
  float root_output;
};
static_assert(sizeof(ResidentParams) == 80, "Metal parameter layout");
static_assert(sizeof(ResidentCandidate) == 64, "Metal candidate layout");
static_assert(sizeof(ResidentTrace) == 80, "Metal trace layout");

const char* kResidentSource = R"METAL(
#include <metal_stdlib>
using namespace metal;
struct Params {
  uint rows, features, leaves, shards, selected_rows, max_depth;
  uint min_data, max_cat_onehot, max_cat_threshold, min_data_group;
  float inv_g, inv_h, l1, l2, min_hessian, min_gain, max_delta, smooth, cat_l2, cat_smooth;
};
struct Feature { uint bins, missing, default_bin, categorical, real_feature; };
struct Leaf { uint begin, count, depth, reserved; float gradient, hessian, output, pad; };
struct Candidate {
  float gain, left_output, right_output, left_hessian, right_hessian;
  uint feature, threshold, default_left, category_bits[8];
};
struct Trace { Candidate candidate; uint leaf, left_count, right_count, valid; };
struct State {
  uint selected, right, small, stopped, step;
  atomic_uint left_count, right_count;
  uint root;
  float root_output;
};
#define BUFFERS \
    device const uchar* bins [[buffer(0)]], \
    device const int* gradients [[buffer(1)]], \
    device const int* hessians [[buffer(2)]], \
    device uint* rows [[buffer(3)]], \
    device uint* scratch [[buffer(4)]], \
    device long2* hist [[buffer(5)]], \
    device long2* partial [[buffer(6)]], \
    device Leaf* leaves [[buffer(7)]], \
    device Candidate* candidates [[buffer(8)]], \
    device Trace* trace [[buffer(9)]], \
    device State* state [[buffer(10)]], \
    device const Feature* features [[buffer(11)]], \
    device const uchar* mask [[buffer(12)]], \
    constant Params& p [[buffer(13)]]

inline float regularized(float g, float l1) {
  return copysign(max(0.0f, abs(g) - l1), g);
}
inline float output_value(float g, float h, float l2, uint count, float parent, constant Params& p) {
  float value = h + l2 > 0 ? -regularized(g, p.l1) / (h + l2) : 0;
  if (p.max_delta > 0) value = clamp(value, -p.max_delta, p.max_delta);
  if (p.smooth > 0) value = (value * float(count) + parent * p.smooth) / (float(count) + p.smooth);
  return value;
}
inline float gain_given_output(float g, float h, float l2, float v, constant Params& p) {
  return -(2.0f * regularized(g, p.l1) * v + (h + l2) * v * v);
}
inline float leaf_gain(float g, float h, float l2, uint count, float parent, constant Params& p) {
  return gain_given_output(g, h, l2, output_value(g, h, l2, count, parent, p), p);
}
inline void consider(thread Candidate& best, float lg, float lh, int lc,
                     Leaf leaf, float l2, float baseline, uint threshold, uint default_left,
                     constant Params& p) {
  const int rc = int(leaf.count) - lc;
  const float rh = leaf.hessian - lh;
  if (lc < int(p.min_data) || rc < int(p.min_data) || lh < p.min_hessian || rh < p.min_hessian) return;
  const float lo = output_value(lg, lh, l2, uint(lc), leaf.output, p);
  const float ro = output_value(leaf.gradient - lg, rh, l2, uint(rc), leaf.output, p);
  const float gain = gain_given_output(lg, lh, l2, lo, p) +
      gain_given_output(leaf.gradient - lg, rh, l2, ro, p) - baseline;
  if (isfinite(gain) && gain > 0 && gain > best.gain) {
    best.gain = gain;
    best.left_output = lo; best.right_output = ro;
    best.left_hessian = lh; best.right_hessian = rh;
    best.threshold = threshold; best.default_left = default_left;
  }
}

// Each chunk is bounded by 256 * 2^20, so 32-bit atomics cannot overflow.
// Long shard totals and long per-leaf histograms retain exact quantized sums.
kernel void resident_histogram(BUFFERS, uint3 group [[threadgroup_position_in_grid]],
                               uint lane [[thread_index_in_threadgroup]]) {
  if (state->stopped) return;
  const uint f = group.x, shard = group.y;
  const Leaf leaf = leaves[state->small];
  if ((!mask[f] && f != 0) || shard * 32768 >= leaf.count) return;
  threadgroup atomic_int gs[256], hs[256];
  long4 total_g(0), total_h(0);
  const uint end = min(leaf.count, (shard + 1) * 32768);
  for (uint start = shard * 32768; start < end; start += 256) {
    for (uint b = lane; b < 256; b += 64) {
      atomic_store_explicit(&gs[b], 0, memory_order_relaxed);
      atomic_store_explicit(&hs[b], 0, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = start + lane; i < min(start + 256, end); i += 64) {
      const uint row = rows[leaf.begin + i];
      const uint bin = bins[ulong(f) * p.rows + row];
      atomic_fetch_add_explicit(&gs[bin], gradients[row], memory_order_relaxed);
      atomic_fetch_add_explicit(&hs[bin], hessians[row], memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint k = 0; k < 4; ++k) {
      total_g[k] += atomic_load_explicit(&gs[lane + 64 * k], memory_order_relaxed);
      total_h[k] += atomic_load_explicit(&hs[lane + 64 * k], memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  for (uint k = 0; k < 4; ++k) {
    partial[(ulong(shard) * p.features + f) * 256 + lane + 64 * k] = long2(total_g[k], total_h[k]);
  }
}

kernel void resident_merge(BUFFERS, uint i [[thread_position_in_grid]]) {
  if (state->stopped || i >= p.features * 256 || (!mask[i / 256] && i / 256 != 0)) return;
  const uint small = state->small;
  const uint shards = (leaves[small].count + 32767) / 32768;
  const ulong size = ulong(p.features) * 256;
  long2 value(0);
  for (uint j = 0; j < shards; ++j) value += partial[ulong(j) * size + i];
  if (!state->root) {
    const uint other = small == state->selected ? state->right : state->selected;
    const long2 parent = hist[ulong(state->selected) * size + i];
    hist[ulong(other) * size + i] = parent - value;
  }
  hist[ulong(small) * size + i] = value;
}

kernel void resident_summary(BUFFERS, uint side [[thread_position_in_grid]]) {
  if (state->stopped || side > 1 || (state->root && side == 1)) return;
  const uint id = side == 0 ? state->selected : state->right;
  long2 sum(0);
  for (uint b = 0; b < features[0].bins; ++b) sum += hist[ulong(id) * p.features * 256 + b];
  leaves[id].gradient = float(sum.x) * p.inv_g;
  leaves[id].hessian = float(sum.y) * p.inv_h;
  if (state->root) {
    float value = -regularized(leaves[id].gradient, p.l1) / (leaves[id].hessian + p.l2);
    if (p.max_delta > 0) value = clamp(value, -p.max_delta, p.max_delta);
    leaves[id].output = value;
    state->root_output = value;
  }
}

kernel void resident_candidates(BUFFERS, uint2 pos [[thread_position_in_grid]]) {
  if (state->stopped || pos.x >= p.features || (state->root && pos.y == 1)) return;
  const uint f = pos.x;
  const uint id = pos.y == 0 ? state->selected : state->right;
  const Leaf leaf = leaves[id];
  const Feature meta = features[f];
  Candidate best = {};
  best.gain = -INFINITY; best.feature = f;
  if (!mask[f] || (!state->root && candidates[ulong(id) * p.features + f].gain <= 0) ||
      leaf.count < 2 * p.min_data || leaf.hessian <= 0 ||
      (p.max_depth > 0 && leaf.depth >= p.max_depth)) {
    candidates[ulong(id) * p.features + f] = best;
    return;
  }
  float g[256], h[256]; int count[256];
  const float factor = float(leaf.count) / leaf.hessian;
  for (uint b = 0; b < meta.bins; ++b) {
    const long2 raw = hist[(ulong(id) * p.features + f) * 256 + b];
    g[b] = float(raw.x) * p.inv_g;
    h[b] = float(raw.y) * p.inv_h;
    count[b] = int(floor(h[b] * factor + 0.5f));
  }
  const float baseline = (p.smooth > 0 && meta.categorical ? gain_given_output(leaf.gradient, leaf.hessian, p.l2,
                                                         leaf.output, p) :
      leaf_gain(leaf.gradient, leaf.hessian, p.l2, leaf.count, leaf.output, p)) + p.min_gain;
  if (meta.categorical) {
    if (meta.bins <= p.max_cat_onehot) {
      for (uint b = 1; b < meta.bins; ++b) {
        const float previous = best.gain;
        consider(best, g[b], h[b], count[b], leaf, p.l2, baseline, b, 0, p);
        if (best.gain > previous) {
          for (uint k = 0; k < 8; ++k) best.category_bits[k] = 0;
          best.category_bits[b / 32] |= 1u << (b % 32);
        }
      }
    } else {
      uint sorted[256]; float keys[256]; uint used = 0;
      for (uint b = 1; b < meta.bins; ++b) {
        if (float(count[b]) < p.cat_smooth) continue;
        const float key = g[b] / (h[b] + p.cat_smooth);
        uint at = used;
        while (at > 0 && key < keys[at - 1]) {
          keys[at] = keys[at - 1]; sorted[at] = sorted[at - 1]; --at;
        }
        keys[at] = key; sorted[at] = b; ++used;
      }
      const uint limit = min(p.max_cat_threshold, (used + 1) / 2);
      for (uint dir = 0; dir < 2; ++dir) {
        float lg = 0, lh = 0; int lc = 0, group_count = 0;
        uint bits[8] = {};
        for (uint j = 0; j < limit; ++j) {
          const uint b = sorted[dir == 0 ? j : used - 1 - j];
          lg += g[b]; lh += h[b]; lc += count[b]; group_count += count[b];
          bits[b / 32] |= 1u << (b % 32);
          if (lc < int(p.min_data) || lh < p.min_hessian) continue;
          if (int(leaf.count) - lc < int(max(p.min_data, p.min_data_group)) ||
              leaf.hessian - lh < p.min_hessian) break;
          if (group_count < int(p.min_data_group)) continue;
          group_count = 0;
          const float previous = best.gain;
          consider(best, lg, lh, lc, leaf, p.l2 + p.cat_l2, baseline, j, 0, p);
          if (best.gain > previous) for (uint k = 0; k < 8; ++k) best.category_bits[k] = bits[k];
        }
      }
    }
  } else {
    // Reverse scan first matches LightGBM's default-left tie preference.
    const bool separate_missing = meta.bins > 2 && meta.missing != 0;
    const uint missing_bin = meta.missing == 1 ? meta.default_bin : meta.bins - 1;
    float rg = 0, rh = 0; int rc = 0;
    const int start = int(meta.bins) - 1 - int(separate_missing && meta.missing == 2);
    for (int b = start; b >= 1; --b) {
      if (separate_missing && meta.missing == 1 && uint(b) == missing_bin) continue;
      rg += g[b]; rh += h[b]; rc += count[b];
      if (rc < int(p.min_data) || rh < p.min_hessian) continue;
      if (int(leaf.count) - rc < int(p.min_data) || leaf.hessian - rh < p.min_hessian) break;
      consider(best, leaf.gradient - rg, leaf.hessian - rh, int(leaf.count) - rc,
               leaf, p.l2, baseline, uint(b - 1), meta.missing == 2 && !separate_missing ? 0 : 1, p);
    }
    if (separate_missing) {
      float lg = 0, lh = 0; int lc = 0;
      for (uint b = 0; b + 1 < meta.bins; ++b) {
        if (meta.missing == 1 && b == missing_bin) continue;
        lg += g[b]; lh += h[b]; lc += count[b];
        if (lc < int(p.min_data) || lh < p.min_hessian) continue;
        if (int(leaf.count) - lc < int(p.min_data) || leaf.hessian - lh < p.min_hessian) break;
        consider(best, lg, lh, lc, leaf, p.l2, baseline, b, 0, p);
      }
    }
  }
  candidates[ulong(id) * p.features + f] = best;
}

kernel void resident_choose(BUFFERS, uint tid [[thread_position_in_grid]]) {
  if (tid != 0 || state->stopped) return;
  Candidate best = {}; best.gain = -INFINITY;
  uint chosen = 0;
  // At equal gains LightGBM first chooses the lower real feature index.
  for (uint leaf = 0; leaf <= state->step; ++leaf) {
    Candidate local = {}; local.gain = -INFINITY;
    for (uint f = 0; f < p.features; ++f) {
      const Candidate c = candidates[ulong(leaf) * p.features + f];
      if (c.gain > local.gain || (c.gain == local.gain &&
          features[c.feature].real_feature < features[local.feature].real_feature)) local = c;
    }
    if (local.gain > best.gain || (local.gain == best.gain &&
        features[local.feature].real_feature < features[best.feature].real_feature)) {
      best = local; chosen = leaf;
    }
  }
  if (!(best.gain > 0) || !isfinite(best.gain)) { state->stopped = 1; return; }
  state->selected = chosen; state->right = state->step + 1;
  atomic_store_explicit(&state->left_count, 0, memory_order_relaxed);
  atomic_store_explicit(&state->right_count, 0, memory_order_relaxed);
  trace[state->step].candidate = best;
  trace[state->step].leaf = chosen;
  trace[state->step].valid = 1;
}

kernel void resident_partition(BUFFERS, uint i [[thread_position_in_grid]]) {
  if (state->stopped) return;
  const Leaf leaf = leaves[state->selected];
  if (i >= leaf.count) return;
  const Candidate c = trace[state->step].candidate;
  const Feature meta = features[c.feature];
  const uint row = rows[leaf.begin + i];
  const uint b = bins[ulong(c.feature) * p.rows + row];
  bool left;
  if (meta.categorical) left = (c.category_bits[b / 32] & (1u << (b % 32))) != 0;
  else if ((meta.missing == 1 && b == meta.default_bin) ||
           (meta.missing == 2 && b == meta.bins - 1)) left = c.default_left != 0;
  else left = b <= c.threshold;
  const uint offset = left ? atomic_fetch_add_explicit(&state->left_count, 1u, memory_order_relaxed) :
                            atomic_fetch_add_explicit(&state->right_count, 1u, memory_order_relaxed);
  scratch[leaf.begin + (left ? offset : leaf.count - 1 - offset)] = row;
}

kernel void resident_copy_rows(BUFFERS, uint i [[thread_position_in_grid]]) {
  if (state->stopped) return;
  const Leaf leaf = leaves[state->selected];
  if (i < leaf.count) rows[leaf.begin + i] = scratch[leaf.begin + i];
}

kernel void resident_finish_split(BUFFERS, uint tid [[thread_position_in_grid]]) {
  if (tid != 0 || state->stopped) return;
  const Leaf parent = leaves[state->selected];
  const uint lc = atomic_load_explicit(&state->left_count, memory_order_relaxed);
  const uint rc = atomic_load_explicit(&state->right_count, memory_order_relaxed);
  if (lc == 0 || rc == 0 || lc + rc != parent.count) { state->stopped = 2; return; }
  const Candidate c = trace[state->step].candidate;
  trace[state->step].left_count = lc; trace[state->step].right_count = rc;
  leaves[state->selected] = {parent.begin, lc, parent.depth + 1, 0, 0, 0, c.left_output, 0};
  leaves[state->right] = {parent.begin + lc, rc, parent.depth + 1, 0, 0, 0, c.right_output, 0};
  // Match SerialTreeLearner's inherited per-feature is_splittable pruning.
  for (uint f = 0; f < p.features; ++f) {
    candidates[ulong(state->right) * p.features + f] = candidates[ulong(state->selected) * p.features + f];
  }
  state->small = lc <= rc ? state->selected : state->right;
  state->root = 0;
  ++state->step;
}
)METAL";

std::string MetalError(NSError* error) {
  return error ? std::string([[error description] UTF8String]) : "unknown Metal error";
}

}  // namespace

class MetalResidentTreeEngine::Impl {
 public:
  explicit Impl(const Dataset* input) : data(input) {}
  const Dataset* data;
  id<MTLDevice> device = nil;
  id<MTLCommandQueue> queue = nil;
  std::array<id<MTLComputePipelineState>, 9> pipelines{};
  std::array<id<MTLBuffer>, 13> buffers{};
  std::vector<ResidentFeature> meta;
  int allocated_leaves = 0;
  bool logged = false;

  void Verify(const Config& c, const ResidentParams& p, DataPartition* partition) {
    const char* verify = std::getenv("LGBM_METAL_RESIDENT_VERIFY");
    if (verify == nullptr || std::strcmp(verify, "1") != 0) return;
    const auto* state = static_cast<const ResidentState*>(buffers[10].contents);
    const auto* trace = static_cast<const ResidentTrace*>(buffers[9].contents);
    const auto* final_leaves = static_cast<const ResidentLeaf*>(buffers[7].contents);
    const auto* final_rows = static_cast<const uint32_t*>(buffers[3].contents);
    const auto* matrix = static_cast<const uint8_t*>(buffers[0].contents);
    const auto* gs = static_cast<const int32_t*>(buffers[1].contents);
    const auto* hs = static_cast<const int32_t*>(buffers[2].contents);
    std::vector<std::vector<data_size_t>> members(c.num_leaves);
    members[0].assign(partition->indices(), partition->indices() + p.selected_rows);
    std::vector<double> outputs(c.num_leaves, state->root_output);
    double max_gain_error = 0;
    for (uint32_t step = 0; step < state->step; ++step) {
      const auto& split = trace[step];
      const auto& best = split.candidate;
      const uint32_t f = best.feature, leaf = split.leaf;
      auto& input = members[leaf];
      std::vector<data_size_t> left(input.size()), right(input.size());
      const data_size_t lc = data->Split(f, meta[f].categorical ? best.category_bits : &best.threshold,
          meta[f].categorical ? 8 : 1, best.default_left != 0, input.data(), input.size(),
          left.data(), right.data());
      CHECK_EQ(lc, split.left_count);
      CHECK_EQ(input.size() - lc, split.right_count);
      std::vector<hist_t> histogram(512, 0.0);
      int64_t total_g = 0, total_h = 0;
      for (const auto row : input) {
        const uint32_t bin = matrix[static_cast<size_t>(f) * p.rows + row];
        histogram[2 * bin] += gs[row]; histogram[2 * bin + 1] += hs[row];
        total_g += gs[row]; total_h += hs[row];
      }
      for (int b = 0; b < 256; ++b) {
        histogram[2 * b] *= p.inv_g; histogram[2 * b + 1] *= p.inv_h;
      }
      FeatureMetainfo fm;
      fm.num_bin = meta[f].bins; fm.missing_type = static_cast<MissingType>(meta[f].missing);
      fm.default_bin = meta[f].default_bin; fm.config = &c;
      fm.bin_type = meta[f].categorical ? BinType::CategoricalBin : BinType::NumericalBin;
      FeatureHistogram reference; reference.Init(histogram.data(), &fm);
      BasicConstraintEntry unconstrained;
      SplitInfo expected;
      reference.FindBestThreshold(total_g * static_cast<double>(p.inv_g),
          total_h * static_cast<double>(p.inv_h), input.size(), &unconstrained, outputs[leaf], &expected);
      const double error = std::abs(expected.gain - best.gain) / std::max(1.0, std::abs(expected.gain));
      max_gain_error = std::max(error, max_gain_error);
      if (!std::isfinite(error) || error > 0.002) {
        Log::Fatal("Metal resident verification: split %u feature %u gain %.9g vs CPU %.9g (relative %.9g)",
                   step, f, best.gain, expected.gain, error);
      }
      left.resize(lc); right.resize(input.size() - lc);
      members[leaf] = std::move(left); members[step + 1] = std::move(right);
      outputs[leaf] = best.left_output; outputs[step + 1] = best.right_output;
    }
    for (uint32_t leaf = 0; leaf <= state->step; ++leaf) {
      auto expected = members[leaf];
      const auto& final = final_leaves[leaf];
      std::vector<uint32_t> got(final_rows + final.begin, final_rows + final.begin + final.count);
      std::sort(got.begin(), got.end()); std::sort(expected.begin(), expected.end());
      CHECK_EQ(got.size(), expected.size());
      for (size_t i = 0; i < got.size(); ++i) CHECK_EQ(got[i], static_cast<uint32_t>(expected[i]));
    }
    Log::Info("Metal resident verification: %u splits, exact CPU row partitions, max relative gain error %.9g",
              state->step, max_gain_error);
  }

  std::string Unsupported(const Config& c) const {
    if (data->num_features() == 0 || data->num_data() == 0) return "empty Dataset";
    for (int f = 0; f < data->num_features(); ++f) {
      if (data->FeatureNumBin(f) > 256) return "a feature has more than 256 bins";
    }
    if (c.use_quantized_grad || c.linear_tree || c.extra_trees || c.feature_fraction_bynode != 1.0 ||
        !c.monotone_constraints.empty() || !c.interaction_constraints_vector.empty() ||
        !c.feature_contri.empty() || c.cegb_penalty_split != 0 ||
        !c.cegb_penalty_feature_lazy.empty() || !c.cegb_penalty_feature_coupled.empty()) {
      return "quantized-grad, linear, random, per-node sampling, constrained or penalized splits";
    }
    if (c.num_leaves > 256) return "more than 256 leaves";
    const uint64_t rows = data->num_data(), features = data->num_features();
    const uint64_t shards = (rows + 32767) / 32768;
    const uint64_t bytes = rows * features + rows * 16 +
        (static_cast<uint64_t>(c.num_leaves) + shards) * features * 256 * 16;
    // This limit covers persistent engine buffers; Dataset/Python storage is additional.
    if (bytes > (uint64_t{4} << 30)) return "resident buffers would exceed 4 GiB";
    return "";
  }

  id<MTLBuffer> Allocate(size_t bytes) {
    if (bytes > device.maxBufferLength) Log::Fatal("Metal resident buffer exceeds device limit");
    id<MTLBuffer> result = [device newBufferWithLength:std::max<size_t>(bytes, 16)
                                            options:MTLResourceStorageModeShared];
    if (result == nil) Log::Fatal("Could not allocate Metal resident buffer (%zu bytes)", bytes);
    return result;
  }

  void Initialize(int num_leaves) {
    if (device == nil) {
      device = MTLCreateSystemDefaultDevice();
      if (device == nil) Log::Fatal("No Metal device for resident training");
      queue = [device newCommandQueue];
      NSError* error = nil;
      MTLCompileOptions* options = [MTLCompileOptions new];
      options.fastMathEnabled = NO;
      id<MTLLibrary> library = [device newLibraryWithSource:[NSString stringWithUTF8String:kResidentSource]
                                                   options:options error:&error];
      if (library == nil) Log::Fatal("Metal resident shader: %s", MetalError(error).c_str());
      const char* names[] = {"resident_histogram", "resident_merge", "resident_summary",
                            "resident_candidates", "resident_choose", "resident_partition",
                            "resident_copy_rows", "resident_finish_split"};
      for (size_t i = 0; i < 8; ++i) {
        id<MTLFunction> function = [library newFunctionWithName:[NSString stringWithUTF8String:names[i]]];
        pipelines[i] = [device newComputePipelineStateWithFunction:function error:&error];
        if (pipelines[i] == nil) Log::Fatal("Metal resident pipeline: %s", MetalError(error).c_str());
      }
      const size_t n = data->num_data(), f = data->num_features();
      buffers[0] = Allocate(n * f);
      for (int i = 1; i <= 4; ++i) buffers[i] = Allocate(n * 4);
      buffers[6] = Allocate(((n + 32767) / 32768) * f * 256 * 16);
      buffers[10] = Allocate(sizeof(ResidentState));
      buffers[11] = Allocate(f * sizeof(ResidentFeature));
      buffers[12] = Allocate(f);
      meta.resize(f);
      auto* matrix = static_cast<uint8_t*>(buffers[0].contents);
      OMP_INIT_EX();
#pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static)
      for (int feature = 0; feature < static_cast<int>(f); ++feature) {
        OMP_LOOP_EX_BEGIN();
        const auto* mapper = data->FeatureBinMapper(feature);
        meta[feature] = {static_cast<uint32_t>(mapper->num_bin()),
                         static_cast<uint32_t>(mapper->missing_type()), mapper->GetDefaultBin(),
                         mapper->bin_type() == BinType::CategoricalBin ? 1u : 0u,
                         static_cast<uint32_t>(data->RealFeatureIndex(feature))};
        std::unique_ptr<BinIterator> iterator(data->FeatureIterator(feature));
        // FeatureIterator expands dense, sparse, bundled and multi-value groups
        // to the actual feature bin, including the omitted most-frequent bin.
        for (data_size_t row = 0; row < data->num_data(); ++row) {
          const uint32_t bin = iterator->Get(row);
          CHECK_LT(bin, meta[feature].bins);
          matrix[static_cast<size_t>(feature) * n + row] = static_cast<uint8_t>(bin);
        }
        OMP_LOOP_EX_END();
      }
      OMP_THROW_EX();
      std::memcpy(buffers[11].contents, meta.data(), f * sizeof(ResidentFeature));
    }
    if (allocated_leaves != num_leaves) {
      const size_t f = data->num_features();
      buffers[5] = Allocate(static_cast<size_t>(num_leaves) * f * 256 * 16);
      buffers[7] = Allocate(num_leaves * sizeof(ResidentLeaf));
      buffers[8] = Allocate(num_leaves * f * sizeof(ResidentCandidate));
      buffers[9] = Allocate(num_leaves * sizeof(ResidentTrace));
      allocated_leaves = num_leaves;
    }
  }

  void Encode(id<MTLCommandBuffer> command, int kernel, const ResidentParams& p,
              MTLSize grid, MTLSize group, bool threadgroups = false) {
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:pipelines[kernel]];
    for (NSUInteger i = 0; i < buffers.size(); ++i) [encoder setBuffer:buffers[i] offset:0 atIndex:i];
    [encoder setBytes:&p length:sizeof(p) atIndex:13];
    if (threadgroups) [encoder dispatchThreadgroups:grid threadsPerThreadgroup:group];
    else [encoder dispatchThreads:grid threadsPerThreadgroup:group];
    [encoder endEncoding];
  }

  Tree* Train(const Config& c, const score_t* gradients, const score_t* hessians,
              const std::vector<int8_t>& mask, DataPartition* partition) {
    @autoreleasepool {
      Initialize(c.num_leaves);
      const uint32_t n = data->num_data(), f = data->num_features();
      const uint32_t selected = partition->leaf_count(0);
      float max_g = 0, max_h = 0;
      int invalid = 0;
#pragma omp parallel for num_threads(OMP_NUM_THREADS()) reduction(max : max_g, max_h, invalid) schedule(static)
      for (data_size_t i = 0; i < static_cast<data_size_t>(n); ++i) {
        if (!std::isfinite(gradients[i]) || !std::isfinite(hessians[i]) || hessians[i] < 0) invalid = 1;
        max_g = std::max(max_g, std::abs(gradients[i]));
        max_h = std::max(max_h, hessians[i]);
      }
      if (invalid) Log::Fatal("Metal resident training requires finite gradients and nonnegative Hessians");
      const float scale_g = max_g > 0 ? std::min(1048576.0f / max_g, 1.0e20f) : 1.0f;
      const float scale_h = max_h > 0 ? std::min(1048576.0f / max_h, 1.0e20f) : 1.0f;
      auto* g = static_cast<int32_t*>(buffers[1].contents);
      auto* h = static_cast<int32_t*>(buffers[2].contents);
#pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static)
      for (data_size_t i = 0; i < static_cast<data_size_t>(n); ++i) {
        g[i] = static_cast<int32_t>(std::nearbyint(gradients[i] * scale_g));
        h[i] = static_cast<int32_t>(std::nearbyint(hessians[i] * scale_h));
      }
      std::memcpy(buffers[3].contents, partition->indices(), selected * sizeof(uint32_t));
      std::memcpy(buffers[12].contents, mask.data(), f);
      std::memset(buffers[7].contents, 0, buffers[7].length);
      std::memset(buffers[9].contents, 0, buffers[9].length);
      auto* leaves = static_cast<ResidentLeaf*>(buffers[7].contents);
      leaves[0].count = selected;
      ResidentState initial{}; initial.root = 1;
      std::memcpy(buffers[10].contents, &initial, sizeof(initial));
      ResidentParams p{n, f, static_cast<uint32_t>(c.num_leaves), (n + 32767) / 32768,
          selected, static_cast<uint32_t>(std::max(0, c.max_depth)),
          static_cast<uint32_t>(c.min_data_in_leaf), static_cast<uint32_t>(c.max_cat_to_onehot),
          static_cast<uint32_t>(c.max_cat_threshold), static_cast<uint32_t>(c.min_data_per_group),
          1.0f / scale_g, 1.0f / scale_h, static_cast<float>(c.lambda_l1), static_cast<float>(c.lambda_l2),
          static_cast<float>(c.min_sum_hessian_in_leaf), static_cast<float>(c.min_gain_to_split),
          static_cast<float>(c.max_delta_step), static_cast<float>(c.path_smooth),
          static_cast<float>(c.cat_l2), static_cast<float>(c.cat_smooth)};
      id<MTLCommandBuffer> command = [queue commandBuffer];
      const MTLSize one = MTLSizeMake(1, 1, 1), threads64 = MTLSizeMake(64, 1, 1);
      const auto hist_and_candidates = [&]() {
        Encode(command, 0, p, MTLSizeMake(f, p.shards, 1), threads64, true);
        Encode(command, 1, p, MTLSizeMake(f * 256, 1, 1), threads64);
        Encode(command, 2, p, MTLSizeMake(2, 1, 1), one);
        Encode(command, 3, p, MTLSizeMake(f, 2, 1), MTLSizeMake(32, 1, 1));
      };
      hist_and_candidates();
      for (int step = 0; step < c.num_leaves - 1; ++step) {
        Encode(command, 4, p, one, one);
        Encode(command, 5, p, MTLSizeMake(selected, 1, 1), threads64);
        Encode(command, 6, p, MTLSizeMake(selected, 1, 1), threads64);
        Encode(command, 7, p, one, one);
        if (step + 1 < c.num_leaves - 1) hist_and_candidates();
      }
      [command commit];
      [command waitUntilCompleted];
      if (command.status == MTLCommandBufferStatusError) {
        Log::Fatal("Metal resident execution failed: %s", MetalError(command.error).c_str());
      }
      const auto* state = static_cast<const ResidentState*>(buffers[10].contents);
      if (state->stopped == 2) Log::Fatal("Metal resident row partition conservation failed");
      Verify(c, p, partition);
      const auto* trace = static_cast<const ResidentTrace*>(buffers[9].contents);
      auto tree = std::unique_ptr<Tree>(new Tree(c.num_leaves, false, false));
      tree->SetLeafOutput(0, state->root_output);
      for (uint32_t step = 0; step < state->step; ++step) {
        CHECK_EQ(trace[step].valid, 1u);
        const auto& split = trace[step];
        const auto& best = split.candidate;
        const int feature = best.feature;
        const auto* mapper = data->FeatureBinMapper(feature);
        if (!meta[feature].categorical) {
          tree->Split(split.leaf, feature, data->RealFeatureIndex(feature), best.threshold,
              data->RealThreshold(feature, best.threshold), best.left_output, best.right_output,
              split.left_count, split.right_count, best.left_hessian, best.right_hessian,
              best.gain + c.min_gain_to_split, mapper->missing_type(), best.default_left != 0);
        } else {
          std::vector<int> values;
          for (uint32_t bin = 0; bin < meta[feature].bins; ++bin) {
            if (best.category_bits[bin / 32] & (1u << (bin % 32))) {
              values.push_back(static_cast<int>(data->RealThreshold(feature, bin)));
            }
          }
          const auto real_bits = Common::ConstructBitset(values.data(), static_cast<int>(values.size()));
          tree->SplitCategorical(split.leaf, feature, data->RealFeatureIndex(feature), best.category_bits, 8,
              real_bits.data(), static_cast<int>(real_bits.size()), best.left_output, best.right_output,
              split.left_count, split.right_count, best.left_hessian, best.right_hessian,
              best.gain + c.min_gain_to_split, mapper->missing_type());
        }
      }
      // Restore final outputs and CPU row partition once per tree. The existing
      // objective renewal, score update, serialization and prediction APIs apply.
      std::vector<data_size_t> begins(tree->num_leaves()), counts(tree->num_leaves());
      for (int leaf = 0; leaf < tree->num_leaves(); ++leaf) {
        tree->SetLeafOutput(leaf, leaves[leaf].output);
        begins[leaf] = leaves[leaf].begin;
        counts[leaf] = leaves[leaf].count;
      }
      partition->SetLeafPartition(static_cast<const data_size_t*>(buffers[3].contents),
                                  selected, begins, counts);
      if (!logged) {
        Log::Info("Metal resident tree: %u feature bins expanded, one command buffer per tree", f);
        logged = true;
      }
      return tree.release();
    }
  }
};

MetalResidentTreeEngine::MetalResidentTreeEngine(const Dataset* data) : impl_(new Impl(data)) {}
MetalResidentTreeEngine::~MetalResidentTreeEngine() = default;
std::string MetalResidentTreeEngine::UnsupportedReason(const Config& config) const {
  return impl_->Unsupported(config);
}
Tree* MetalResidentTreeEngine::Train(const Config& config, const score_t* gradients,
                                     const score_t* hessians, const std::vector<int8_t>& mask,
                                     DataPartition* partition) {
  return impl_->Train(config, gradients, hessians, mask, partition);
}

}  // namespace LightGBM
