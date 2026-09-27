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
  uint32_t skip_default, sparse_scan, binary_tiles, chunk_rows, stage_gradients;
};
struct ResidentFeature {
  uint32_t bins, missing, default_bin, categorical, real_feature, most_freq, sparse_start, sparse_count, binary_slot;
};
struct ResidentLeaf {
  uint32_t begin, count, depth, reserved;
  float gradient, hessian, output, pad;
  int64_t gradient_q, hessian_q;
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
  uint32_t histogram_groups[3], partition_groups[3], binary_groups[3], stable_groups[3], staging_groups[3];
};
static_assert(sizeof(ResidentParams) == 100, "Metal parameter layout");
static_assert(sizeof(ResidentCandidate) == 64, "Metal candidate layout");
static_assert(sizeof(ResidentTrace) == 80, "Metal trace layout");

const char* kResidentSource = R"METAL(
#include <metal_stdlib>
using namespace metal;
struct Params {
  uint rows, features, leaves, shards, selected_rows, max_depth;
  uint min_data, max_cat_onehot, max_cat_threshold, min_data_group;
  float inv_g, inv_h, l1, l2, min_hessian, min_gain, max_delta, smooth, cat_l2, cat_smooth;
  uint skip_default, sparse_scan, binary_tiles, chunk_rows, stage_gradients;
};
struct Feature { uint bins, missing, default_bin, categorical, real_feature, most_freq, sparse_start, sparse_count, binary_slot; };
struct Leaf {
  uint begin, count, depth, reserved; float gradient, hessian, output, pad;
  long gradient_q, hessian_q;
};
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
  uint histogram_groups[3], partition_groups[3], binary_groups[3], stable_groups[3], staging_groups[3];
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
    constant Params& p [[buffer(13)]], \
    device Candidate* leaf_best [[buffer(14)]], \
    device uint* row_leaf [[buffer(15)]], \
    device const uint* sparse_rows [[buffer(16)]], \
    device const uint* binary_features [[buffer(17)]], \
    device const ulong* binary_bits [[buffer(18)]], \
    device const ulong* binary_masks [[buffer(19)]], \
    device uint* partition_prefix [[buffer(20)]], \
    device int2* selected_gradients [[buffer(21)]]

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

kernel void resident_stage_gradients(BUFFERS, uint i [[thread_position_in_grid]]) {
  if (state->stopped) return;
  const Leaf leaf = leaves[state->small];
  if (i < leaf.count) {
    const uint row = rows[leaf.begin + i];
    selected_gradients[leaf.begin + i] = int2(gradients[row], hessians[row]);
  }
}

// Each chunk is bounded by 1024 * 2^20, so 32-bit atomics cannot overflow.
// Long shard totals and long per-leaf histograms retain exact quantized sums.
kernel void resident_histogram(BUFFERS, uint3 group [[threadgroup_position_in_grid]],
                               uint lane [[thread_index_in_threadgroup]]) {
  if (state->stopped) return;
  const uint f = group.x, shard = group.y;
  const Leaf leaf = leaves[state->small];
  const Feature meta = features[f];
  if (p.binary_tiles > 0 && meta.binary_slot != UINT_MAX) return;
  const bool sparse = p.sparse_scan && p.skip_default && f != 0 &&
      meta.sparse_count > 0 && meta.sparse_count < leaf.count;
  const uint work_count = sparse ? meta.sparse_count : leaf.count;
  if ((!mask[f] && f != 0) || shard * 32768 >= work_count) return;
  threadgroup atomic_int gs[256], hs[256];
  long4 total_g(0), total_h(0);
  const uint end = min(work_count, (shard + 1) * 32768);
  for (uint start = shard * 32768; start < end; start += p.chunk_rows) {
    for (uint b = lane; b < meta.bins; b += 64) {
      atomic_store_explicit(&gs[b], 0, memory_order_relaxed);
      atomic_store_explicit(&hs[b], 0, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = start + lane; i < min(start + p.chunk_rows, end); i += 64) {
      const uint row = sparse ? sparse_rows[meta.sparse_start + i] : rows[leaf.begin + i];
      if (sparse && row_leaf[row] != state->small) continue;
      const uint bin = bins[ulong(f) * p.rows + row];
      if (p.skip_default && f != 0 && bin == features[f].most_freq) continue;
      const int2 gh = p.stage_gradients && !sparse ? selected_gradients[leaf.begin + i] :
                                                  int2(gradients[row], hessians[row]);
      atomic_fetch_add_explicit(&gs[bin], gh.x, memory_order_relaxed);
      atomic_fetch_add_explicit(&hs[bin], gh.y, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint k = 0; k < 4; ++k) {
      if (lane + 64 * k >= meta.bins) break;
      total_g[k] += atomic_load_explicit(&gs[lane + 64 * k], memory_order_relaxed);
      total_h[k] += atomic_load_explicit(&hs[lane + 64 * k], memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  for (uint k = 0; k < 4; ++k) {
    if (lane + 64 * k >= meta.bins) break;
    partial[(ulong(shard) * p.features + f) * 256 + lane + 64 * k] = long2(total_g[k], total_h[k]);
  }
}

// Pack 64 binary features into one row word. One workgroup then builds all
// 64 non-default histograms while loading each selected gradient only once.
kernel void resident_binary_histogram(BUFFERS, uint3 group [[threadgroup_position_in_grid]],
                                      uint lane [[thread_index_in_threadgroup]]) {
  if (state->stopped || group.x >= p.binary_tiles) return;
  const Leaf leaf = leaves[state->small];
  const uint tile = group.x, shard = group.y;
  if (shard * 32768 >= leaf.count) return;
  const ulong active = binary_masks[tile];
  threadgroup atomic_int gs[64], hs[64];
  long total_g = 0, total_h = 0;
  const uint end = min(leaf.count, (shard + 1) * 32768);
  for (uint start = shard * 32768; start < end; start += p.chunk_rows) {
    atomic_store_explicit(&gs[lane], 0, memory_order_relaxed);
    atomic_store_explicit(&hs[lane], 0, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = start + lane; i < min(start + p.chunk_rows, end); i += 64) {
      const uint row = rows[leaf.begin + i];
      ulong bits = binary_bits[ulong(tile) * p.rows + row] & active;
      if (bits != 0) {
        const int2 gh = p.stage_gradients ? selected_gradients[leaf.begin + i] :
                                          int2(gradients[row], hessians[row]);
        while (bits != 0) {
          const uint bin = uint(ctz(bits));
          bits &= bits - 1;
          atomic_fetch_add_explicit(&gs[bin], gh.x, memory_order_relaxed);
          atomic_fetch_add_explicit(&hs[bin], gh.y, memory_order_relaxed);
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    total_g += atomic_load_explicit(&gs[lane], memory_order_relaxed);
    total_h += atomic_load_explicit(&hs[lane], memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  const uint f = binary_features[tile * 64 + lane];
  if (f != UINT_MAX) {
    const ulong offset = (ulong(shard) * p.features + f) * 256;
    partial[offset + features[f].most_freq] = long2(0);
    partial[offset + 1 - features[f].most_freq] = long2(total_g, total_h);
  }
}

kernel void resident_merge(BUFFERS, uint i [[thread_position_in_grid]]) {
  if (state->stopped || i >= p.features * 256 || (!mask[i / 256] && i / 256 != 0)) return;
  if (i % 256 >= features[i / 256].bins) return;
  const uint small = state->small;
  const Feature meta = features[i / 256];
  const bool sparse = p.sparse_scan && p.skip_default && i / 256 != 0 &&
      !(p.binary_tiles > 0 && meta.binary_slot != UINT_MAX) &&
      meta.sparse_count > 0 && meta.sparse_count < leaves[small].count;
  const uint shards = ((sparse ? meta.sparse_count : leaves[small].count) + 32767) / 32768;
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
  leaves[id].gradient_q = sum.x; leaves[id].hessian_q = sum.y;
  if (state->root) {
    float value = -regularized(leaves[id].gradient, p.l1) / (leaves[id].hessian + p.l2);
    if (p.max_delta > 0) value = clamp(value, -p.max_delta, p.max_delta);
    leaves[id].output = value;
    state->root_output = value;
  }
}

kernel void resident_fix_default(BUFFERS, uint2 pos [[thread_position_in_grid]]) {
  if (!p.skip_default || state->stopped || pos.x == 0 || pos.x >= p.features ||
      !mask[pos.x] || (state->root && pos.y == 1)) return;
  const uint f = pos.x;
  const uint id = pos.y == 0 ? state->selected : state->right;
  const uint default_bin = features[f].most_freq;
  const ulong offset = (ulong(id) * p.features + f) * 256;
  long2 other(0);
  for (uint b = 0; b < features[f].bins; ++b) if (b != default_bin) other += hist[offset + b];
  hist[offset + default_bin] = long2(leaves[id].gradient_q, leaves[id].hessian_q) - other;
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

inline bool better_candidate(Candidate a, Candidate b, device const Feature* features) {
  return a.gain > b.gain || (a.gain == b.gain &&
         features[a.feature].real_feature < features[b.feature].real_feature);
}
kernel void resident_leaf_best(BUFFERS, uint side [[threadgroup_position_in_grid]],
                               uint lane [[thread_index_in_threadgroup]]) {
  if (state->stopped || (state->root && side == 1)) return;
  const uint id = side == 0 ? state->selected : state->right;
  Candidate best = {}; best.gain = -INFINITY;
  for (uint f = lane; f < p.features; f += 64) {
    const Candidate c = candidates[ulong(id) * p.features + f];
    if (better_candidate(c, best, features)) best = c;
  }
  threadgroup Candidate shared[64];
  shared[lane] = best;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint stride = 32; stride > 0; stride /= 2) {
    if (lane < stride && better_candidate(shared[lane + stride], shared[lane], features)) {
      shared[lane] = shared[lane + stride];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (lane == 0) leaf_best[id] = shared[0];
}

kernel void resident_choose(BUFFERS, uint tid [[thread_position_in_grid]]) {
  if (tid != 0 || state->stopped) return;
  Candidate best = {}; best.gain = -INFINITY;
  uint chosen = 0;
  // At equal gains LightGBM first chooses the lower real feature index.
  for (uint leaf = 0; leaf <= state->step; ++leaf) {
    const Candidate local = leaf_best[leaf];
    if (local.gain > best.gain || (local.gain == best.gain &&
        features[local.feature].real_feature < features[best.feature].real_feature)) {
      best = local; chosen = leaf;
    }
  }
  if (!(best.gain > 0) || !isfinite(best.gain)) {
    state->stopped = 1;
    state->partition_groups[0] = 1; state->histogram_groups[1] = 1;
    state->binary_groups[1] = 1;
    state->stable_groups[0] = 1;
    return;
  }
  state->selected = chosen; state->right = state->step + 1;
  state->partition_groups[0] = (leaves[chosen].count + 63) / 64;
  state->stable_groups[0] = (leaves[chosen].count + 255) / 256;
  atomic_store_explicit(&state->left_count, 0, memory_order_relaxed);
  atomic_store_explicit(&state->right_count, 0, memory_order_relaxed);
  trace[state->step].candidate = best;
  trace[state->step].leaf = chosen;
  trace[state->step].valid = 1;
}

inline bool route_bin(uint b, Feature meta, Candidate c) {
  if (meta.categorical) return (c.category_bits[b / 32] & (1u << (b % 32))) != 0;
  if ((meta.missing == 1 && b == meta.default_bin) ||
      (meta.missing == 2 && b == meta.bins - 1)) return c.default_left != 0;
  return b <= c.threshold;
}

kernel void resident_partition_counts(BUFFERS, uint i [[thread_position_in_grid]],
                                      uint lane [[thread_index_in_threadgroup]],
                                      uint simd_lane [[thread_index_in_simdgroup]],
                                      uint simd_group [[simdgroup_index_in_threadgroup]],
                                      uint group [[threadgroup_position_in_grid]]) {
  if (state->stopped) return;
  const Leaf leaf = leaves[state->selected];
  const Candidate c = trace[state->step].candidate;
  uint left = 0;
  if (i < leaf.count) {
    const uint row = rows[leaf.begin + i];
    left = uint(route_bin(bins[ulong(c.feature) * p.rows + row], features[c.feature], c));
  }
  const uint sum = simd_sum(left);
  threadgroup uint sums[8];
  if (simd_lane == 0) sums[simd_group] = sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lane == 0) {
    uint total = 0;
    for (uint j = 0; j < 8; ++j) total += sums[j];
    partition_prefix[group] = total;
  }
}

kernel void resident_partition_prefix(BUFFERS, uint tid [[thread_position_in_grid]]) {
  if (state->stopped || tid != 0) return;
  const uint count = leaves[state->selected].count;
  const uint groups = (count + 255) / 256;
  uint total = 0;
  for (uint group = 0; group < groups; ++group) {
    const uint value = partition_prefix[group];
    partition_prefix[group] = total;
    total += value;
  }
  atomic_store_explicit(&state->left_count, total, memory_order_relaxed);
  atomic_store_explicit(&state->right_count, count - total, memory_order_relaxed);
}

kernel void resident_partition_stable(BUFFERS, uint i [[thread_position_in_grid]],
                                      uint lane [[thread_index_in_threadgroup]],
                                      uint simd_lane [[thread_index_in_simdgroup]],
                                      uint simd_group [[simdgroup_index_in_threadgroup]],
                                      uint group [[threadgroup_position_in_grid]]) {
  if (state->stopped) return;
  const Leaf leaf = leaves[state->selected];
  const Candidate c = trace[state->step].candidate;
  uint left = 0, row = 0;
  if (i < leaf.count) {
    row = rows[leaf.begin + i];
    left = uint(route_bin(bins[ulong(c.feature) * p.rows + row], features[c.feature], c));
  }
  uint before = simd_prefix_exclusive_sum(left);
  const uint sum = simd_sum(left);
  threadgroup uint sums[8];
  if (simd_lane == 0) sums[simd_group] = sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint j = 0; j < simd_group; ++j) before += sums[j];
  before += partition_prefix[group];
  if (i < leaf.count) {
    const uint total_left = atomic_load_explicit(&state->left_count, memory_order_relaxed);
    scratch[leaf.begin + (left ? before : total_left + i - before)] = row;
    row_leaf[row] = left ? state->selected : state->right;
  }
}

kernel void resident_partition(BUFFERS, uint i [[thread_position_in_grid]]) {
  if (state->stopped) return;
  const Leaf leaf = leaves[state->selected];
  if (i >= leaf.count) return;
  const Candidate c = trace[state->step].candidate;
  const Feature meta = features[c.feature];
  const uint row = rows[leaf.begin + i];
  const uint b = bins[ulong(c.feature) * p.rows + row];
  const bool left = route_bin(b, meta, c);
  row_leaf[row] = left ? state->selected : state->right;
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
  state->histogram_groups[1] = (min(lc, rc) + 32767) / 32768;
  state->binary_groups[1] = state->histogram_groups[1];
  state->staging_groups[0] = (min(lc, rc) + 255) / 256;
  state->root = 0;
  ++state->step;
}
)METAL";

std::string MetalError(NSError* error) {
  return error ? std::string([[error description] UTF8String]) : "unknown Metal error";
}

bool EnvironmentFlag(const char* name, bool fallback) {
  const char* value = std::getenv(name);
  if (value == nullptr) return fallback;
  if (std::strcmp(value, "0") == 0) return false;
  if (std::strcmp(value, "1") == 0) return true;
  Log::Fatal("%s must be 0 or 1", name);
  return fallback;
}

}  // namespace

class MetalResidentTreeEngine::Impl {
 public:
  explicit Impl(const Dataset* input) : data(input) {
    const char* profile = std::getenv("LGBM_METAL_PROFILE");
    profile_enabled = profile != nullptr && std::strcmp(profile, "1") == 0;
    indirect = EnvironmentFlag("LGBM_METAL_RESIDENT_INDIRECT", true);
    profile_stages = EnvironmentFlag("LGBM_METAL_RESIDENT_PROFILE_STAGES", false);
    stable_partition = EnvironmentFlag("LGBM_METAL_RESIDENT_STABLE", true);
    skip_default = EnvironmentFlag("LGBM_METAL_RESIDENT_SKIP_DEFAULT", true);
    sparse_scan = EnvironmentFlag("LGBM_METAL_RESIDENT_SPARSE", true);
    stage_gradients = EnvironmentFlag("LGBM_METAL_RESIDENT_STAGE", true);
    binary_enabled = EnvironmentFlag("LGBM_METAL_RESIDENT_BINARY", true);
    verify_enabled = EnvironmentFlag("LGBM_METAL_RESIDENT_VERIFY", false);
    const char* chunk = std::getenv("LGBM_METAL_RESIDENT_CHUNK");
    if (chunk != nullptr) {
      if (std::strcmp(chunk, "256") == 0) chunk_rows = 256;
      else if (std::strcmp(chunk, "512") == 0) chunk_rows = 512;
      else if (std::strcmp(chunk, "1024") != 0) Log::Fatal("LGBM_METAL_RESIDENT_CHUNK must be 256, 512 or 1024");
    }
  }
  ~Impl() {
    if (profile_enabled) {
      Log::Info("Metal resident profile: trees=%u initialize=%.6fs quantize=%.6fs encode=%.6fs inflight=%.6fs gpu=%.6fs materialize=%.6fs",
                trained, initialize_seconds, quantize_seconds, encode_seconds, inflight_seconds,
                gpu_seconds, materialize_seconds);
    }
    if (profile_stages) {
      Log::Info("Metal resident SERIALIZED stage diagnostic: hist=%.6fs merge=%.6fs summary=%.6fs candidates=%.6fs choose=%.6fs partition=%.6fs copy=%.6fs finish=%.6fs default=%.6fs",
          stage_seconds[0], stage_seconds[1], stage_seconds[2], stage_seconds[3], stage_seconds[4],
          stage_seconds[5], stage_seconds[6], stage_seconds[7], stage_seconds[8]);
      Log::Info("Metal resident SERIALIZED leaf candidate reduction: %.6fs", stage_seconds[9]);
      Log::Info("Metal resident SERIALIZED binary histogram: %.6fs", stage_seconds[10]);
      Log::Info("Metal resident SERIALIZED stable partition: count=%.6fs prefix=%.6fs scatter=%.6fs",
                stage_seconds[11], stage_seconds[12], stage_seconds[13]);
      Log::Info("Metal resident SERIALIZED gradient staging: %.6fs", stage_seconds[14]);
    }
  }
  const Dataset* data;
  id<MTLDevice> device = nil;
  id<MTLCommandQueue> queue = nil;
  std::array<id<MTLComputePipelineState>, 15> pipelines{};
  std::array<id<MTLBuffer>, 13> buffers{};
  id<MTLBuffer> leaf_best_buffer = nil;
  id<MTLBuffer> row_leaf_buffer = nil;
  id<MTLBuffer> sparse_rows_buffer = nil;
  id<MTLBuffer> binary_features_buffer = nil, binary_bits_buffer = nil, binary_masks_buffer = nil;
  uint32_t binary_tiles = 0;
  id<MTLBuffer> partition_prefix_buffer = nil;
  id<MTLBuffer> selected_gradients_buffer = nil;
  std::vector<ResidentFeature> meta;
  int allocated_leaves = 0;
  bool logged = false;
  bool profile_enabled = false;
  bool indirect = true;
  bool profile_stages = false;
  bool stable_partition = true;
  bool skip_default = true, sparse_scan = true, stage_gradients = true, binary_enabled = true;
  bool verify_enabled = false;
  uint32_t chunk_rows = 1024;
  std::array<double, 15> stage_seconds{};
  uint32_t trained = 0;
  double initialize_seconds = 0, quantize_seconds = 0, encode_seconds = 0;
  double inflight_seconds = 0, gpu_seconds = 0, materialize_seconds = 0;
  using Clock = std::chrono::steady_clock;
  static double Elapsed(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
  }

  void Verify(const Config& c, const ResidentParams& p, DataPartition* partition) {
    if (!verify_enabled) return;
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
    const uint64_t bytes = rows * features * 2 + rows * 28 +
        (static_cast<uint64_t>(c.num_leaves) + shards) * features * 256 * 16 +
        static_cast<uint64_t>(c.num_leaves) * features * sizeof(ResidentCandidate);
    // Include a conservative bound for sparse indices and mirror temporaries.
    // Dataset/Python storage is additional.
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
                            "resident_copy_rows", "resident_finish_split", "resident_fix_default", "resident_leaf_best",
                            "resident_binary_histogram", "resident_partition_counts",
                            "resident_partition_prefix", "resident_partition_stable", "resident_stage_gradients"};
      for (size_t i = 0; i < 15; ++i) {
        id<MTLFunction> function = [library newFunctionWithName:[NSString stringWithUTF8String:names[i]]];
        pipelines[i] = [device newComputePipelineStateWithFunction:function error:&error];
        if (pipelines[i] == nil) Log::Fatal("Metal resident pipeline: %s", MetalError(error).c_str());
      }
      if (pipelines[11].threadExecutionWidth != 32 || pipelines[13].threadExecutionWidth != 32 ||
          pipelines[11].maxTotalThreadsPerThreadgroup < 256 || pipelines[13].maxTotalThreadsPerThreadgroup < 256) {
        stable_partition = false;
      }
      const size_t n = data->num_data(), f = data->num_features();
      buffers[0] = Allocate(n * f);
      for (int i = 1; i <= 4; ++i) buffers[i] = Allocate(n * 4);
      buffers[6] = Allocate(((n + 32767) / 32768) * f * 256 * 16);
      buffers[10] = Allocate(sizeof(ResidentState));
      buffers[11] = Allocate(f * sizeof(ResidentFeature));
      buffers[12] = Allocate(f);
      row_leaf_buffer = Allocate(n * sizeof(uint32_t));
      partition_prefix_buffer = Allocate(((n + 255) / 256) * sizeof(uint32_t));
      selected_gradients_buffer = Allocate(n * 2 * sizeof(int32_t));
      meta.resize(f);
      std::vector<std::vector<uint32_t>> sparse_indices(f);
      auto* matrix = static_cast<uint8_t*>(buffers[0].contents);
      OMP_INIT_EX();
#pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static)
      for (int feature = 0; feature < static_cast<int>(f); ++feature) {
        OMP_LOOP_EX_BEGIN();
        const auto* mapper = data->FeatureBinMapper(feature);
        meta[feature] = {static_cast<uint32_t>(mapper->num_bin()),
                         static_cast<uint32_t>(mapper->missing_type()), mapper->GetDefaultBin(),
                         mapper->bin_type() == BinType::CategoricalBin ? 1u : 0u,
                         static_cast<uint32_t>(data->RealFeatureIndex(feature)), mapper->GetMostFreqBin(), 0, 0,
                         std::numeric_limits<uint32_t>::max()};
        const bool sparse = feature != 0 && mapper->sparse_rate() >= 0.9;
        std::unique_ptr<BinIterator> iterator(data->FeatureIterator(feature));
        // FeatureIterator expands dense, sparse, bundled and multi-value groups
        // to the actual feature bin, including the omitted most-frequent bin.
        for (data_size_t row = 0; row < data->num_data(); ++row) {
          const uint32_t bin = iterator->Get(row);
          CHECK_LT(bin, meta[feature].bins);
          matrix[static_cast<size_t>(feature) * n + row] = static_cast<uint8_t>(bin);
          if (sparse && bin != mapper->GetMostFreqBin()) sparse_indices[feature].push_back(row);
        }
        if (sparse_indices[feature].size() > n / 10) sparse_indices[feature].clear();
        OMP_LOOP_EX_END();
      }
      OMP_THROW_EX();
      size_t sparse_size = 0;
      for (const auto& column : sparse_indices) sparse_size += column.size();
      CHECK_LE(sparse_size, static_cast<size_t>(std::numeric_limits<uint32_t>::max()));
      sparse_rows_buffer = Allocate(sparse_size * sizeof(uint32_t));
      auto* sparse_dest = static_cast<uint32_t*>(sparse_rows_buffer.contents);
      uint32_t offset = 0;
      for (size_t feature = 0; feature < f; ++feature) {
        meta[feature].sparse_start = offset;
        meta[feature].sparse_count = static_cast<uint32_t>(sparse_indices[feature].size());
        std::copy(sparse_indices[feature].begin(), sparse_indices[feature].end(), sparse_dest + offset);
        offset += meta[feature].sparse_count;
      }
      std::vector<uint32_t> binary_features;
      if (binary_enabled) {
        for (size_t feature = 1; feature < f; ++feature) {
          if (meta[feature].bins == 2) {
            meta[feature].binary_slot = static_cast<uint32_t>(binary_features.size());
            binary_features.push_back(static_cast<uint32_t>(feature));
          }
        }
      }
      binary_tiles = static_cast<uint32_t>((binary_features.size() + 63) / 64);
      binary_features.resize(binary_tiles * 64, std::numeric_limits<uint32_t>::max());
      binary_features_buffer = Allocate(binary_features.size() * sizeof(uint32_t));
      std::memcpy(binary_features_buffer.contents, binary_features.data(), binary_features.size() * sizeof(uint32_t));
      binary_bits_buffer = Allocate(static_cast<size_t>(binary_tiles) * n * sizeof(uint64_t));
      binary_masks_buffer = Allocate(binary_tiles * sizeof(uint64_t));
      auto* bits = static_cast<uint64_t*>(binary_bits_buffer.contents);
      std::memset(bits, 0, binary_bits_buffer.length);
#pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static)
      for (int tile = 0; tile < static_cast<int>(binary_tiles); ++tile) {
        uint64_t* tile_bits = bits + static_cast<size_t>(tile) * n;
        for (int lane = 0; lane < 64; ++lane) {
          const uint32_t feature = binary_features[tile * 64 + lane];
          if (feature == std::numeric_limits<uint32_t>::max()) break;
          if (!sparse_indices[feature].empty()) {
            for (const auto row : sparse_indices[feature]) tile_bits[row] |= uint64_t{1} << lane;
          } else {
            for (size_t row = 0; row < n; ++row) {
              if (matrix[feature * n + row] != meta[feature].most_freq) tile_bits[row] |= uint64_t{1} << lane;
            }
          }
        }
      }
      std::memcpy(buffers[11].contents, meta.data(), f * sizeof(ResidentFeature));
    }
    if (allocated_leaves != num_leaves) {
      const size_t f = data->num_features();
      buffers[5] = Allocate(static_cast<size_t>(num_leaves) * f * 256 * 16);
      buffers[7] = Allocate(num_leaves * sizeof(ResidentLeaf));
      buffers[8] = Allocate(num_leaves * f * sizeof(ResidentCandidate));
      buffers[9] = Allocate(num_leaves * sizeof(ResidentTrace));
      leaf_best_buffer = Allocate(num_leaves * sizeof(ResidentCandidate));
      allocated_leaves = num_leaves;
    }
  }

  void Encode(id<MTLCommandBuffer> __strong& command, int kernel, const ResidentParams& p,
              MTLSize grid, MTLSize group, bool threadgroups = false, int indirect_offset = -1) {
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:pipelines[kernel]];
    for (NSUInteger i = 0; i < buffers.size(); ++i) [encoder setBuffer:buffers[i] offset:0 atIndex:i];
    [encoder setBytes:&p length:sizeof(p) atIndex:13];
    [encoder setBuffer:leaf_best_buffer offset:0 atIndex:14];
    [encoder setBuffer:row_leaf_buffer offset:0 atIndex:15];
    [encoder setBuffer:sparse_rows_buffer offset:0 atIndex:16];
    [encoder setBuffer:binary_features_buffer offset:0 atIndex:17];
    [encoder setBuffer:binary_bits_buffer offset:0 atIndex:18];
    [encoder setBuffer:binary_masks_buffer offset:0 atIndex:19];
    [encoder setBuffer:partition_prefix_buffer offset:0 atIndex:20];
    [encoder setBuffer:selected_gradients_buffer offset:0 atIndex:21];
    if (indirect_offset >= 0 && indirect) {
      [encoder dispatchThreadgroupsWithIndirectBuffer:buffers[10] indirectBufferOffset:indirect_offset
                                threadsPerThreadgroup:group];
    } else if (threadgroups) [encoder dispatchThreadgroups:grid threadsPerThreadgroup:group];
    else [encoder dispatchThreads:grid threadsPerThreadgroup:group];
    [encoder endEncoding];
    if (profile_stages) {
      [command commit]; [command waitUntilCompleted];
      if (command.status == MTLCommandBufferStatusError) {
        Log::Fatal("Metal resident stage diagnostic failed: %s", MetalError(command.error).c_str());
      }
      stage_seconds[kernel] += command.GPUEndTime - command.GPUStartTime;
      command = [queue commandBuffer];
    }
  }

  Tree* Train(const Config& c, const score_t* gradients, const score_t* hessians,
              const std::vector<int8_t>& mask, DataPartition* partition) {
    @autoreleasepool {
      auto checkpoint = Clock::now();
      Initialize(c.num_leaves);
      if (profile_enabled) initialize_seconds += Elapsed(checkpoint);
      checkpoint = Clock::now();
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
      auto* row_leaf = static_cast<uint32_t*>(row_leaf_buffer.contents);
      std::fill(row_leaf, row_leaf + n, std::numeric_limits<uint32_t>::max());
      for (uint32_t i = 0; i < selected; ++i) row_leaf[partition->indices()[i]] = 0;
      std::memcpy(buffers[12].contents, mask.data(), f);
      auto* binary_masks = static_cast<uint64_t*>(binary_masks_buffer.contents);
      std::fill(binary_masks, binary_masks + binary_tiles, 0);
      for (uint32_t feature = 0; feature < f; ++feature) {
        const uint32_t slot = meta[feature].binary_slot;
        if (mask[feature] && slot != std::numeric_limits<uint32_t>::max()) {
          binary_masks[slot / 64] |= uint64_t{1} << (slot % 64);
        }
      }
      std::memset(buffers[7].contents, 0, buffers[7].length);
      std::memset(buffers[9].contents, 0, buffers[9].length);
      auto* leaves = static_cast<ResidentLeaf*>(buffers[7].contents);
      leaves[0].count = selected;
      ResidentState initial{}; initial.root = 1;
      initial.histogram_groups[0] = f;
      initial.histogram_groups[1] = (selected + 32767) / 32768;
      initial.histogram_groups[2] = 1;
      initial.partition_groups[0] = (selected + 63) / 64;
      initial.partition_groups[1] = initial.partition_groups[2] = 1;
      initial.binary_groups[0] = binary_tiles;
      initial.binary_groups[1] = initial.histogram_groups[1];
      initial.binary_groups[2] = 1;
      initial.stable_groups[0] = (selected + 255) / 256;
      initial.stable_groups[1] = initial.stable_groups[2] = 1;
      initial.staging_groups[0] = initial.stable_groups[0];
      initial.staging_groups[1] = initial.staging_groups[2] = 1;
      std::memcpy(buffers[10].contents, &initial, sizeof(initial));
      ResidentParams p{n, f, static_cast<uint32_t>(c.num_leaves), (n + 32767) / 32768,
          selected, static_cast<uint32_t>(std::max(0, c.max_depth)),
          static_cast<uint32_t>(c.min_data_in_leaf), static_cast<uint32_t>(c.max_cat_to_onehot),
          static_cast<uint32_t>(c.max_cat_threshold), static_cast<uint32_t>(c.min_data_per_group),
          1.0f / scale_g, 1.0f / scale_h, static_cast<float>(c.lambda_l1), static_cast<float>(c.lambda_l2),
          static_cast<float>(c.min_sum_hessian_in_leaf), static_cast<float>(c.min_gain_to_split),
          static_cast<float>(c.max_delta_step), static_cast<float>(c.path_smooth),
          static_cast<float>(c.cat_l2), static_cast<float>(c.cat_smooth),
          static_cast<uint32_t>(skip_default), static_cast<uint32_t>(sparse_scan), binary_tiles, chunk_rows,
          static_cast<uint32_t>(stage_gradients)};
      if (!p.skip_default) p.binary_tiles = 0;
      id<MTLCommandBuffer> command = [queue commandBuffer];
      if (profile_enabled) quantize_seconds += Elapsed(checkpoint);
      checkpoint = Clock::now();
      const MTLSize one = MTLSizeMake(1, 1, 1), threads64 = MTLSizeMake(64, 1, 1);
      const auto hist_and_candidates = [&]() {
        if (p.stage_gradients) {
          Encode(command, 14, p, MTLSizeMake((selected + 255) / 256, 1, 1), MTLSizeMake(256, 1, 1), true,
                 offsetof(ResidentState, staging_groups));
        }
        Encode(command, 0, p, MTLSizeMake(f, p.shards, 1), threads64, true,
               offsetof(ResidentState, histogram_groups));
        if (p.binary_tiles > 0) {
          Encode(command, 10, p, MTLSizeMake(p.binary_tiles, p.shards, 1), threads64, true,
                 offsetof(ResidentState, binary_groups));
        }
        Encode(command, 1, p, MTLSizeMake(f * 256, 1, 1), threads64);
        Encode(command, 2, p, MTLSizeMake(2, 1, 1), one);
        Encode(command, 8, p, MTLSizeMake(f, 2, 1), MTLSizeMake(32, 1, 1));
        Encode(command, 3, p, MTLSizeMake(f, 2, 1), MTLSizeMake(32, 1, 1));
        Encode(command, 9, p, MTLSizeMake(2, 1, 1), threads64, true);
      };
      hist_and_candidates();
      for (int step = 0; step < c.num_leaves - 1; ++step) {
        Encode(command, 4, p, one, one);
        if (stable_partition) {
          const MTLSize grid = MTLSizeMake((selected + 255) / 256, 1, 1);
          Encode(command, 11, p, grid, MTLSizeMake(256, 1, 1), true, offsetof(ResidentState, stable_groups));
          Encode(command, 12, p, one, one);
          Encode(command, 13, p, grid, MTLSizeMake(256, 1, 1), true, offsetof(ResidentState, stable_groups));
        } else {
          Encode(command, 5, p, MTLSizeMake(selected, 1, 1), threads64, false,
                 offsetof(ResidentState, partition_groups));
        }
        Encode(command, 6, p, MTLSizeMake(selected, 1, 1), threads64, false,
               offsetof(ResidentState, partition_groups));
        Encode(command, 7, p, one, one);
        if (step + 1 < c.num_leaves - 1) hist_and_candidates();
      }
      if (profile_enabled) encode_seconds += Elapsed(checkpoint);
      checkpoint = Clock::now();
      [command commit];
      [command waitUntilCompleted];
      if (profile_enabled) {
        inflight_seconds += Elapsed(checkpoint);
        gpu_seconds += command.GPUEndTime - command.GPUStartTime;
      }
      checkpoint = Clock::now();
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
      if (profile_enabled) materialize_seconds += Elapsed(checkpoint);
      ++trained;
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
