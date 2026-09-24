"""Small synthetic correctness checks for the experimental Metal backend.

Writes one machine-readable JSON report and exits nonzero on a failed case.
No external dataset is read and no row-level predictions are saved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, roc_auc_score

import lightgbm as lgb
from lightgbm.libpath import _find_lib_path


def compare(
    name: str, x_train, y_train, x_held, y_held, *, rounds: int, objective: str = "binary", extra: dict | None = None
) -> dict:
    dataset = lgb.Dataset(x_train, label=y_train, free_raw_data=False, params={"max_bin": 63})
    dataset.construct()
    params = {
        "objective": objective,
        "metric": "binary_logloss" if objective == "binary" else "l2",
        "verbosity": 1,
        "num_threads": 2,
        "num_leaves": 15,
        "min_data_in_leaf": 30,
        "max_bin": 63,
        "seed": 20260923,
        "force_col_wise": True,
    }
    params.update(extra or {})
    runs = {}
    predictions = {}
    for device in ("cpu", "metal"):
        start = time.perf_counter()
        model = lgb.train({**params, "device_type": device}, dataset, num_boost_round=rounds)
        fit_seconds = time.perf_counter() - start
        prediction = model.predict(x_held)
        predictions[device] = prediction
        metric = roc_auc_score(y_held, prediction) if objective == "binary" else mean_squared_error(y_held, prediction)
        runs[device] = {
            "fit_seconds_observational": round(fit_seconds, 6),
            "metric": float(metric),
            "trees": model.num_trees(),
        }
        if name == "dense_numeric" and device == "metal":
            with tempfile.TemporaryDirectory(prefix="lgbm-metal-test-") as tmp:
                path = Path(tmp) / "model.txt"
                model.save_model(str(path))
                reloaded = lgb.Booster(model_file=str(path))
                reload_error = float(np.max(np.abs(reloaded.predict(x_held) - prediction)))
                if reload_error > 1e-12:
                    raise AssertionError(f"model reload changed predictions: {reload_error}")
    difference = np.abs(predictions["cpu"] - predictions["metal"])
    max_difference = float(difference.max())
    metric_difference = abs(runs["cpu"]["metric"] - runs["metal"]["metric"])
    if max_difference > 1e-3:
        raise AssertionError(f"{name}: prediction difference {max_difference}")
    if metric_difference > (1e-4 if objective == "binary" else 1e-5):
        raise AssertionError(f"{name}: metric difference {metric_difference}")
    result = {
        "case": name,
        "status": "PASS",
        "train_rows": len(y_train),
        "rounds": rounds,
        "cpu": runs["cpu"],
        "metal": runs["metal"],
        "max_prediction_difference": max_difference,
        "mean_prediction_difference": float(difference.mean()),
    }
    print("CASE " + json.dumps(result), flush=True)
    return result


def compare_overlap() -> dict:
    """Check that concurrent and serialized CPU/GPU work produce one model."""
    rng = np.random.default_rng(119)
    x = rng.normal(size=(10_000, 24)).astype(np.float32)
    x[:, 16:] *= rng.random((10_000, 8)) < 0.06
    y = (x[:, 0] + 0.7 * x[:, 1] * x[:, 2] + 0.3 * x[:, 16] > 0).astype(np.int8)
    dataset = lgb.Dataset(x[:8_000], label=y[:8_000], free_raw_data=False, params={"max_bin": 63})
    dataset.construct()
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "verbosity": 1,
        "device_type": "metal",
        "num_threads": 2,
        "num_leaves": 15,
        "min_data_in_leaf": 30,
        "max_bin": 63,
        "seed": 20260923,
        "force_col_wise": True,
        "feature_fraction": 0.8,
    }
    variable = "LGBM_METAL_DISABLE_OVERLAP"
    original = os.environ.pop(variable, None)
    try:
        overlapping = lgb.train(params, dataset, num_boost_round=20)
        os.environ[variable] = "1"
        serial = lgb.train(params, dataset, num_boost_round=20)
    finally:
        if original is None:
            os.environ.pop(variable, None)
        else:
            os.environ[variable] = original
    model_equal = overlapping.model_to_string() == serial.model_to_string()
    difference = float(np.max(np.abs(overlapping.predict(x[8_000:]) - serial.predict(x[8_000:]))))
    if not model_equal or difference > 1e-12:
        raise AssertionError(f"overlap changed model or predictions: {difference}")
    result = {
        "case": "cpu_gpu_overlap",
        "status": "PASS",
        "train_rows": 8_000,
        "rounds": 20,
        "model_text_equal": model_equal,
        "max_prediction_difference": difference,
    }
    print("CASE " + json.dumps(result), flush=True)
    return result


def run_cases() -> list[dict]:
    cases = []
    rng = np.random.default_rng(321)
    x = rng.normal(size=(16_000, 12)).astype(np.float32)
    y = (x[:, 0] + 0.8 * x[:, 1] * x[:, 2] + 0.5 * x[:, 3] > 0.2).astype(np.int8)
    cases.append(compare("dense_numeric", x[:12_000], y[:12_000], x[12_000:], y[12_000:], rounds=30))

    rng = np.random.default_rng(99)
    n = 16_000
    numeric = rng.normal(size=(n, 10)).astype(np.float32)
    numeric[rng.random((n, 10)) < 0.12] = np.nan
    sparse = rng.normal(size=(n, 6)).astype(np.float32)
    sparse[rng.random((n, 6)) < 0.9] = 0
    categories = rng.integers(0, 30, size=n)
    mixed = pd.DataFrame(np.column_stack([numeric, sparse]), columns=[f"f{i}" for i in range(16)])
    mixed["category"] = pd.Categorical(categories)
    y = (
        (
            np.nan_to_num(numeric[:, 0])
            + 0.6 * np.nan_to_num(numeric[:, 1]) * np.nan_to_num(numeric[:, 2])
            + 0.5 * (categories % 3 == 0)
            + 0.3 * sparse[:, 0]
        )
        > 0.2
    ).astype(np.int8)
    cases.append(
        compare(
            "mixed_missing_category_sparse_bagging",
            mixed.iloc[:12_000],
            y[:12_000],
            mixed.iloc[12_000:],
            y[12_000:],
            rounds=20,
            extra={"feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1},
        )
    )

    rng = np.random.default_rng(7)
    x = rng.normal(size=(130_000, 16)).astype(np.float32)
    y = (x[:, 0] + x[:, 1] * x[:, 2] + 0.1 * rng.normal(size=len(x)) > 0).astype(np.int8)
    cases.append(compare("multiple_histogram_shards", x[:100_000], y[:100_000], x[100_000:], y[100_000:], rounds=10))

    rng = np.random.default_rng(77)
    x = rng.normal(size=(8_000, 6)).astype(np.float32)
    y = (1.5 * x[:, 0] - 0.2 * x[:, 1] + rng.normal(scale=0.1, size=len(x))).astype(np.float32)
    cases.append(
        compare(
            "constant_hessian_cpu_fallback",
            x[:6_000],
            y[:6_000],
            x[6_000:],
            y[6_000:],
            rounds=10,
            objective="regression",
        )
    )
    cases.append(compare_overlap())
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("metal_validation.json"))
    args = parser.parse_args()
    report = {
        "scope": "synthetic_model_level_correctness_only",
        "machine": platform.machine(),
        "library_sha256": hashlib.sha256(Path(_find_lib_path()[0]).read_bytes()).hexdigest(),
    }
    try:
        report["cases"] = run_cases()
        report["status"] = "PASS"
    except Exception as error:
        report["status"] = "FAIL"
        report["error"] = f"{type(error).__name__}: {error}"
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        raise
    report["note"] = "Case times are observational, not a speed benchmark."
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"PASS {args.output}", flush=True)


if __name__ == "__main__":
    main()
