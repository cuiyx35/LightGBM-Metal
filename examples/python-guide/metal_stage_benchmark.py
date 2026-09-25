"""Compare the legacy and adaptive Metal histogram paths on generated data.

No external dataset or labels are read. This is a Metal-to-Metal comparison,
not a CPU speed benchmark. Full scale requires substantial unified memory.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python-package"))

from metal_synthetic_benchmark import check_idle, make_data  # noqa: E402

import lightgbm as lgb  # noqa: E402
from lightgbm.libpath import _find_lib_path  # noqa: E402


def digest_model(model: lgb.Booster) -> str:
    return hashlib.sha256(model.model_to_string().encode()).hexdigest()


def digest_prediction(prediction: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(prediction, dtype="<f8").tobytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full-scale", action="store_true", help="Use 3.3 million fit rows, 512 features, and 100 trees"
    )
    parser.add_argument("--output", type=Path, default=Path("metal_stage_benchmark.json"))
    args = parser.parse_args()
    if os.environ.get("LGBM_METAL_STAGE_SELECTED") is not None:
        parser.error("Unset LGBM_METAL_STAGE_SELECTED; this script controls it")
    if any(
        os.environ.get(key) is not None
        for key in (
            "LGBM_METAL_PROFILE",
            "LGBM_METAL_FORCE_CPU",
            "LGBM_METAL_ROWS_PER_CHUNK",
            "LGBM_METAL_ROWS_PER_SHARD",
            "LGBM_METAL_THREADS_PER_GROUP",
            "LGBM_METAL_MIN_LEAF_ROWS",
            "LGBM_METAL_COMPACT_GROUPS",
            "LGBM_METAL_DISABLE_OVERLAP",
            "LGBM_METAL_COMPARE_HIST",
            "LGBM_METAL_VERIFY_QUANTIZED_GROUP",
            "LGBM_METAL_VERIFY_QUANTIZED_DISPATCH",
        )
    ):
        parser.error("Unset other Metal diagnostic overrides for this comparison")
    if not Path(lgb.__file__).resolve().is_relative_to(REPO / "python-package/lightgbm"):
        raise RuntimeError("Load this checkout's lightgbm Python package")
    if Path(_find_lib_path()[0]).resolve() != (REPO / "lib_lightgbm.dylib").resolve():
        raise RuntimeError("Load this checkout's Metal-enabled native library")
    config = (
        {
            "train_rows": 3_300_000,
            "held_rows": 200_000,
            "features": 512,
            "dense_features": 150,
            "rounds": 100,
            "threads": 6,
        }
        if args.full_scale
        else {
            "train_rows": 30_000,
            "held_rows": 5_000,
            "features": 40,
            "dense_features": 12,
            "rounds": 10,
            "threads": 2,
        }
    )
    check_idle()
    x, labels = make_data(
        config["train_rows"] + config["held_rows"], config["features"], config["dense_features"], 20260924, 0.05
    )
    dataset = lgb.Dataset(
        x[: config["train_rows"]], label=labels[: config["train_rows"]], params={"max_bin": 127}, free_raw_data=True
    )
    dataset.construct()
    held = x[config["train_rows"] :]
    held_labels = labels[config["train_rows"] :]
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "verbosity": -1,
        "learning_rate": 0.025,
        "num_leaves": 63,
        "min_data_in_leaf": 35,
        "lambda_l2": 4.0,
        "feature_fraction": 0.8,
        "max_bin": 127,
        "num_threads": config["threads"],
        "seed": 20260924,
        "force_col_wise": True,
        "device_type": "metal",
    }
    order = ["legacy", "adaptive", "adaptive", "legacy"]
    runs: list[dict] = []
    first_prediction: np.ndarray | None = None
    max_prediction_difference = 0.0
    try:
        for mode in order:
            check_idle()
            os.environ["LGBM_METAL_STAGE_SELECTED"] = "2" if mode == "adaptive" else "0"
            start = time.perf_counter()
            model = lgb.train(params, dataset, num_boost_round=config["rounds"])
            fit_seconds = time.perf_counter() - start
            prediction = model.predict(held, num_threads=config["threads"])
            if first_prediction is None:
                first_prediction = prediction.copy()
            else:
                max_prediction_difference = max(
                    max_prediction_difference, float(np.max(np.abs(prediction - first_prediction)))
                )
            row = {
                "mode": mode,
                "fit_seconds": round(fit_seconds, 6),
                "average_precision": float(average_precision_score(held_labels, prediction)),
                "auc": float(roc_auc_score(held_labels, prediction)),
                "model_sha256": digest_model(model),
                "prediction_sha256": digest_prediction(prediction),
            }
            runs.append(row)
            print("RUN " + json.dumps(row), flush=True)
            del model, prediction
            gc.collect()
    finally:
        os.environ.pop("LGBM_METAL_STAGE_SELECTED", None)
    grouped = {mode: [row for row in runs if row["mode"] == mode] for mode in ("legacy", "adaptive")}
    if any(
        len({row["model_sha256"] for row in group}) != 1 or len({row["prediction_sha256"] for row in group}) != 1
        for group in grouped.values()
    ):
        raise RuntimeError("A repeated mode produced different model or prediction hashes")
    medians = {mode: statistics.median(row["fit_seconds"] for row in group) for mode, group in grouped.items()}
    report = {
        "status": "PASS",
        "scope": "generated_data_only",
        "configuration": config,
        "seed": 20260924,
        "run_order": order,
        "runs": runs,
        "median_fit_seconds": medians,
        "legacy_to_adaptive_fit_ratio": medians["legacy"] / medians["adaptive"],
        "same_model_across_modes": grouped["legacy"][0]["model_sha256"] == grouped["adaptive"][0]["model_sha256"],
        "same_prediction_across_modes": grouped["legacy"][0]["prediction_sha256"]
        == grouped["adaptive"][0]["prediction_sha256"],
        "max_prediction_difference": max_prediction_difference,
        "external_data_read": False,
        "row_predictions_written": False,
        "limitations": [
            "One Apple Silicon machine, one generated seed, two runs per mode.",
            "No one-tree warmup; data generation and Dataset construction are excluded from fit times.",
            "Thermals and background load may affect even interleaved timings.",
            "This compares two Metal algorithms, not CPU with Metal or application model quality.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(
        "RESULT "
        + json.dumps(
            {
                "legacy_to_adaptive_fit_ratio": report["legacy_to_adaptive_fit_ratio"],
                "same_model_across_modes": report["same_model_across_modes"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
