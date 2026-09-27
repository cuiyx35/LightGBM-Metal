# GPU-resident tree experiment

[简体中文](Metal-Resident.md) · **English** · [Project home](../README.en.md)

## Enable

With this fork built using <code>USE_METAL</code>, set the environment variable before creating the learner:

~~~python
import os
import lightgbm as lgb

os.environ["LGBM_METAL_RESIDENT"] = "1"
model = lgb.train(
    {"objective": "binary", "device_type": "metal", "num_threads": 6},
    train_set,
    num_boost_round=100,
)
~~~

Unset the variable or use <code>0</code> for the default CPU/GPU hybrid path. Environment variables are read when creating the learner; they cannot select different modes for models created concurrently in one process. Prediction, saving and loading use standard LightGBM interfaces.

## Work on GPU and CPU

A supported tree computes histograms, split candidates, best-leaf selection and row partitioning within one Metal command buffer. The GPU keeps the leaf queue and histograms. Only the smaller child is scanned; the larger child's histogram is obtained by subtracting from its parent. At the end of each tree, the CPU materializes a regular LightGBM model and imports the final row partition.

Objectives, per-iteration gradients/Hessians, sampling, training-score updates and model objects still use the CPU trainer. On the first fit, the CPU expands actual Dataset feature bins into a GPU byte matrix, including bundled, sparse and multi-value groups. Binary features use packed words; sparse non-default rows have a separate index.

## Support and fallback

| Item | Current scope |
| --- | --- |
| Bins and leaves | Up to 256 bins per feature and 256 leaves |
| Splits | Numerical, categorical one-hot and subsets, None/Zero/NaN missing routing |
| Constraints and regularization | Minimum count/Hessian/gain, maximum depth, L1/L2, max_delta_step, path_smooth and categorical regularization |
| Tested training flows | Binary classification, regression, multiclass, regression_l1 leaf renewal, bagging, GOSS, per-tree feature sampling, resizing leaf pools, continuation from a saved model |
| Explicit fallback | Bin/leaf limits exceeded, monotone/interaction constraints, feature_contri, CEGB, extra_trees, per-node feature sampling, forced splits, upstream quantized gradients and linear_tree |
| Memory | Fallback when a conservative estimate of expanded bins and work buffers exceeds 4 GiB; Dataset, Python data and temporary storage are additional |

Fallback logs contain <code>resident tree fallback:</code>; active resident execution logs contain <code>resident tree:</code>. The existing learner handles unsupported configurations with their constraints preserved. <code>LGBM_METAL_FORCE_CPU=1</code> takes precedence.

## Numerics and validation

Each tree quantizes gradients/Hessians to a maximum absolute integer of <code>2^20</code>. A chunk of at most 1,024 rows accumulates using 32-bit threadgroup atomics, with an absolute bound of <code>2^30</code>. Shard totals, leaf histograms and subtraction use 64-bit integers. Split scoring uses GPU single precision; nearly tied candidates may differ from CPU results.

~~~bash
PYTHONPATH="$PWD/python-package" python3 examples/python-guide/metal_resident_validate.py \
  --output metal_resident_validation.json
~~~

Native diagnostics replay CPU routing for every selected GPU split, compare final row sets, and call the CPU FeatureHistogram to verify the best gain of the selected feature. The script also checks cached training scores, normal prediction, model reload, generated holdout metrics and fallback behavior. Diagnostic timings are excluded from performance claims.

Directly replacing a mixed sparse Dataset triggered an empty-child assertion in the CPU baseline and has not been accepted. When changing data, prefer a new Booster with the saved model as <code>init_model</code>. Numerical Dataset replacement is covered. Generated-data validation does not establish application quality.

## Performance comparison

~~~bash
PYTHONPATH="$PWD/python-package" python3 examples/python-guide/metal_resident_benchmark.py \
  --output metal_resident_quick.json
# 3.3 million training rows, 200k generated holdout rows, 512 features, 100 trees, 6 threads
PYTHONPATH="$PWD/python-package" python3 examples/python-guide/metal_resident_benchmark.py \
  --full-scale --output metal_resident_full.json
~~~

The script shares a Dataset, warms each backend with one tree, then runs resident→hybrid→CPU→CPU→hybrid→resident. Fit includes native initialization and mirroring, excluding shared data generation and Dataset construction; prediction time is separate. JSON records source/library hashes, model/prediction hashes, AP/AUC/log loss, prediction differences and load. <code>PASS</code> means generated-data quality and repeatability checks passed, not that resident training was faster.

## Development

The implementation is in <code>src/treelearner/metal_resident_tree.{h,mm}</code>. <code>MetalTreeLearner::Train()</code> selects the path; <code>DataPartition::SetLeafPartition()</code> imports the final row pool. Changes must check fallback, parameter resets, histogram conservation and the CPU model interface.

| Diagnostic variable | Purpose |
| --- | --- |
| <code>LGBM_METAL_RESIDENT_VERIFY=1</code> | Native CPU split/row-set verification |
| <code>LGBM_METAL_PROFILE=1</code> | Cumulative initialization, quantization, encoding, GPU and materialization times |
| <code>LGBM_METAL_RESIDENT_PROFILE_STAGES=1</code> | Submit and wait per kernel; changes execution and is diagnostic only |
| <code>LGBM_METAL_RESIDENT_INDIRECT=0</code> | Disable GPU indirect dispatch |
| <code>LGBM_METAL_RESIDENT_SKIP_DEFAULT=0</code> | Disable most-frequent-bin omission and conservation reconstruction |
| <code>LGBM_METAL_RESIDENT_SPARSE=0</code> | Disable sparse row-list scanning |
| <code>LGBM_METAL_RESIDENT_BINARY=0</code> | Disable binary feature packing |
| <code>LGBM_METAL_RESIDENT_STABLE=0</code> | Use atomic rather than stable partitioning |
| <code>LGBM_METAL_RESIDENT_STAGE=0</code> | Disable selected-leaf gradient staging |
| <code>LGBM_METAL_RESIDENT_CHUNK=256&#124;512&#124;1024</code> | Atomic chunk size; default 1,024 |

Boolean resident diagnostics accept only <code>0</code> or <code>1</code>. Performance scripts reject external Metal overrides to keep backend labels meaningful. Objective-C++ sources also need OpenMP compiler flags for parallel CPU bin expansion and quantization.
