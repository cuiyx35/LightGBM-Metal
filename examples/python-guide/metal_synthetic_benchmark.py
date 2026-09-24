"""Reproducible, public-data-only CPU versus Metal LightGBM benchmark.

The default geometry is a large single-fold workload. Every row, feature,
and label is generated here from a fixed seed. No external dataset is read.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import resource
import statistics
import subprocess
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def check_idle() -> None:
    """Optional local guard; unset for a fully standalone public benchmark."""
    marker = os.environ.get("LGBM_BENCHMARK_GUARD_PREFIX")
    if not marker:
        return
    output = subprocess.check_output(["ps", "-axo", "pid=,command="], text=True)
    matches = []
    for line in output.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) == 2 and int(parts[0]) != os.getpid() and marker in parts[1]:
            matches.append(parts[0])
    if matches:
        raise RuntimeError(f"Competing process detected ({', '.join(matches)})")


def make_data(rows: int, features: int, dense_features: int, seed: int,
              prevalence: float) -> tuple[np.ndarray, np.ndarray]:
    """Use uint8 storage to keep large public benchmark generation bounded."""
    rng = np.random.default_rng(seed)
    x = np.empty((rows, features), dtype=np.uint8)
    for j in range(dense_features):
        x[:, j] = rng.integers(0, 127, size=rows, dtype=np.uint8)
    for j in range(dense_features, features):
        # Rare nonzero columns allow LightGBM to exercise feature bundling.
        x[:, j] = (rng.random(rows) < 0.015).astype(np.uint8)
    score = (x[:, 0].astype(np.float32) - 63) / 28
    score += ((x[:, 1].astype(np.float32) - 63) *
              (x[:, 2].astype(np.float32) - 63)) / 1800
    score += (x[:, min(20, dense_features - 1)].astype(np.float32) - 63) / 65
    if dense_features < features:
        score += 0.45 * x[:, dense_features].astype(np.float32)
    score += rng.standard_normal(rows).astype(np.float32) * 0.8
    threshold = np.quantile(score, 1 - prevalence)
    labels = (score >= threshold).astype(np.uint8)
    return x, labels


def digest_model(model: lgb.Booster) -> str:
    return hashlib.sha256(model.model_to_string().encode()).hexdigest()


def digest_predictions(prediction: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(
        prediction, dtype="<f8").tobytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-rows", type=int, default=3_300_000)
    parser.add_argument("--held-rows", type=int, default=800_000)
    parser.add_argument("--features", type=int, default=512)
    parser.add_argument("--dense-features", type=int, default=150)
    parser.add_argument("--rounds", type=int, default=500)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--repeats", type=int, choices=(1, 2), default=2)
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--positive-rate", type=float, default=0.05)
    parser.add_argument("--output", type=Path,
                        default=Path("metal_synthetic_benchmark.json"))
    args = parser.parse_args()
    if (args.train_rows < 100 or args.held_rows < 100 or args.features < 3 or
            not 3 <= args.dense_features <= args.features or args.rounds < 1 or
            args.threads < 1 or not 0 < args.positive_rate < 0.5):
        parser.error("Invalid workload dimensions or class prevalence")
    for key in ("LGBM_METAL_PROFILE", "LGBM_METAL_FORCE_CPU",
                "LGBM_METAL_DISABLE_OVERLAP", "LGBM_METAL_COMPARE_HIST"):
        if os.getenv(key):
            parser.error(f"Unset diagnostic/override variable {key}")

    check_idle()
    start = time.perf_counter()
    x, labels = make_data(args.train_rows + args.held_rows, args.features,
                          args.dense_features, args.seed, args.positive_rate)
    generate_seconds = time.perf_counter() - start
    print(f"DATA_READY rows={len(x)} features={x.shape[1]} "
          f"storage_gib={x.nbytes / 2**30:.3f} seconds={generate_seconds:.3f}",
          flush=True)
    check_idle()

    params = {
        "objective": "binary", "metric": "binary_logloss", "verbosity": 1,
        "learning_rate": 0.025, "num_leaves": 63, "min_data_in_leaf": 35,
        "lambda_l2": 4.0, "feature_fraction": 0.8, "max_bin": 127,
        "num_threads": args.threads, "seed": args.seed, "force_col_wise": True,
    }
    start = time.perf_counter()
    dataset = lgb.Dataset(x[:args.train_rows], label=labels[:args.train_rows],
                          params={"max_bin": 127}, free_raw_data=True)
    dataset.construct()
    construct_seconds = time.perf_counter() - start
    print(f"DATASET_READY seconds={construct_seconds:.3f}", flush=True)
    check_idle()

    warmups = []
    for backend in ("cpu", "metal"):
        start = time.perf_counter()
        model = lgb.train({**params, "device_type": backend}, dataset,
                          num_boost_round=1)
        warmups.append({"backend": backend,
                        "seconds": round(time.perf_counter() - start, 6)})
        del model
        gc.collect()
        check_idle()

    order = (["metal", "cpu", "cpu", "metal"] if args.repeats == 2
             else ["metal", "cpu"])
    runs = []
    representative = {}
    for index, backend in enumerate(order):
        check_idle()
        start = time.perf_counter()
        model = lgb.train({**params, "device_type": backend}, dataset,
                          num_boost_round=args.rounds)
        fit_seconds = time.perf_counter() - start
        if model.num_trees() != args.rounds:
            raise RuntimeError(f"{backend} trained {model.num_trees()} trees")
        start = time.perf_counter()
        prediction = model.predict(x[args.train_rows:],
                                   num_threads=args.threads)
        predict_seconds = time.perf_counter() - start
        run = {
            "index": index, "backend": backend,
            "fit_seconds": round(fit_seconds, 6),
            "predict_seconds": round(predict_seconds, 6),
            "average_precision": float(average_precision_score(
                labels[args.train_rows:], prediction)),
            "auc": float(roc_auc_score(labels[args.train_rows:], prediction)),
            "model_sha256": digest_model(model),
            "prediction_sha256": digest_predictions(prediction),
            "trees": model.num_trees(),
        }
        if backend not in representative:
            representative[backend] = prediction
        else:
            if (run["model_sha256"] != next(r["model_sha256"] for r in runs
                                            if r["backend"] == backend) or
                    run["prediction_sha256"] != next(
                        r["prediction_sha256"] for r in runs
                        if r["backend"] == backend)):
                raise RuntimeError(f"{backend} repeated result changed")
        runs.append(run)
        print("RUN " + json.dumps({k: run[k] for k in (
            "index", "backend", "fit_seconds", "predict_seconds",
            "average_precision", "auc")}), flush=True)
        del model
        if backend in representative and representative[backend] is not prediction:
            del prediction
        gc.collect()
        check_idle()

    diff = np.abs(representative["metal"] - representative["cpu"])
    medians = {backend: {
        "fit_seconds": statistics.median(
            r["fit_seconds"] for r in runs if r["backend"] == backend),
        "fit_plus_predict_seconds": statistics.median(
            r["fit_seconds"] + r["predict_seconds"] for r in runs
            if r["backend"] == backend),
    } for backend in ("cpu", "metal")}
    report = {
        "status": "PASS", "scope": "fully_synthetic_scale_benchmark",
        "lightgbm_version": getattr(lgb, "__version__", "source-checkout"),
        "machine": platform.machine(),
        "configuration": {
            "train_rows": args.train_rows, "held_rows": args.held_rows,
            "features": args.features, "dense_features": args.dense_features,
            "rare_binary_features": args.features - args.dense_features,
            "rounds": args.rounds, "threads": args.threads,
            "seed": args.seed, "positive_rate": args.positive_rate,
            "data_dtype": "uint8", "max_bin": 127,
            "parameters": params,
        },
        "generate_seconds_shared": round(generate_seconds, 6),
        "dataset_construct_seconds_shared": round(construct_seconds, 6),
        "warmups_not_timed": warmups, "run_order": order, "runs": runs,
        "median": medians,
        "cpu_over_metal_fit_speedup": (
            medians["cpu"]["fit_seconds"] / medians["metal"]["fit_seconds"]),
        "cpu_over_metal_fit_plus_predict_speedup": (
            medians["cpu"]["fit_plus_predict_seconds"] /
            medians["metal"]["fit_plus_predict_seconds"]),
        "metal_vs_cpu_prediction": {
            "mean_absolute_difference": float(diff.mean()),
            "maximum_absolute_difference": float(diff.max()),
            "count_above_0_001": int(np.count_nonzero(diff > 0.001)),
            "count_above_0_01": int(np.count_nonzero(diff > 0.01)),
        },
        "peak_process_rss_gib": round(resource.getrusage(
            resource.RUSAGE_SELF).ru_maxrss / 2**30, 3),
        "limitations": [
            "Only workload geometry is similar to an application workload; "
            "feature and label distributions are entirely synthetic.",
            "One machine and one seed; timing varies with system load.",
            "The shared generation and Dataset construction times are excluded "
            "from per-backend training times.",
            "The synthetic quality metrics cannot establish application quality.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"PASS {args.output}", flush=True)


if __name__ == "__main__":
    main()
