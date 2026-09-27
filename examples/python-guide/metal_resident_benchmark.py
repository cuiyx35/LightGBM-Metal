"""Compare CPU, hybrid Metal and opt-in resident trees on generated data.

Default: a small smoke workload. --full-scale selects a 3.3-million-row,
100-tree comparison. This script never reads application datasets.
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

import numpy as np
from metal_synthetic_benchmark import check_idle, digest_model, digest_predictions, make_data
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

import lightgbm as lgb
from lightgbm.libpath import _find_lib_path

REPO = Path(__file__).resolve().parents[2]


class CaptureLogger:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def info(self, message: str) -> None:
        self.messages.append(message)

    def warning(self, message: str) -> None:
        self.messages.append(message)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-scale", action="store_true")
    parser.add_argument("--rounds", type=int, help="Override tree count within the selected workload")
    parser.add_argument("--seed", type=int, default=20260927)
    parser.add_argument("--output", type=Path, default=Path("metal_resident_benchmark.json"))
    args = parser.parse_args()
    if args.rounds is not None and args.rounds < 1:
        parser.error("--rounds must be positive")
    for name in os.environ:
        if name.startswith("LGBM_METAL_"):
            parser.error(f"Unset diagnostic/override variable {name}")
    if not Path(lgb.__file__).resolve().is_relative_to(REPO / "python-package/lightgbm"):
        raise RuntimeError("Use this checkout's lightgbm Python package")
    library = Path(_find_lib_path()[0]).resolve()
    if library != (REPO / "lib_lightgbm.dylib").resolve():
        raise RuntimeError("Use this checkout's Metal-enabled native library")
    train_rows, held_rows, features, dense, rounds, threads = (
        (3_300_000, 200_000, 512, 150, 100, 6) if args.full_scale else (30_000, 5_000, 40, 12, 15, 2)
    )
    rounds = args.rounds or rounds
    params = {
        "objective": "binary",
        "verbosity": 1,
        "learning_rate": 0.05,
        "lambda_l2": 4.0,
        "num_leaves": 63,
        "max_bin": 127,
        "feature_fraction": 0.8,
        "num_threads": threads,
        "seed": args.seed,
        "force_col_wise": True,
    }
    report = {
        "status": "RUNNING",
        "scope": "generated_data_resident_comparison",
        "configuration": {
            "train_rows": train_rows,
            "held_rows": held_rows,
            "features": features,
            "dense_features": dense,
            "rounds": rounds,
            "seed": args.seed,
            "parameters": params,
        },
        "machine": {"architecture": platform.machine(), "macos": platform.mac_ver()[0]},
        "native_library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
        "source_files_sha256": {
            name: hashlib.sha256((REPO / name).read_bytes()).hexdigest()
            for name in (
                "src/treelearner/metal_resident_tree.mm",
                "src/treelearner/metal_tree_learner.mm",
                "CMakeLists.txt",
            )
        },
        "runs": [],
        "warmups": [],
        "limitations": [
            "Generated distributions cannot establish application model quality.",
            "Fit includes native initialization/mirroring and excludes the shared Dataset construction.",
            "Peak RSS is the cumulative process maximum, not per-backend memory use.",
            "One machine; temperature and GPU hardware counters are not measured.",
            "Optional LGBM_BENCHMARK_GUARD_PREFIX checks competing process command lines before each fit.",
        ],
    }
    try:
        report["source_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
        report["source_dirty"] = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO, text=True))
    except (OSError, subprocess.CalledProcessError):
        report["source_commit"] = None
        report["source_dirty"] = None
    logger = CaptureLogger()
    lgb.register_logger(logger)
    predictions = {}

    def save() -> None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=4) + "\n")

    def fit(backend, trees):
        check_idle()
        logger.messages.clear()
        os.environ["LGBM_METAL_RESIDENT"] = "1" if backend == "resident" else "0"
        start = time.perf_counter()
        model = lgb.train(
            {**params, "device_type": "cpu" if backend == "cpu" else "metal"}, dataset, num_boost_round=trees
        )
        elapsed = time.perf_counter() - start
        if backend == "resident" and (
            not any("resident tree:" in m for m in logger.messages)
            or any("resident tree fallback:" in m for m in logger.messages)
        ):
            raise RuntimeError("Expected resident execution without fallback")
        if model.num_trees() != trees:
            raise RuntimeError(f"{backend}: expected {trees} trees, got {model.num_trees()}")
        return model, elapsed

    try:
        check_idle()
        report["load_average_before"] = list(os.getloadavg())
        start = time.perf_counter()
        x, y = make_data(train_rows + held_rows, features, dense, args.seed, 0.05)
        report["generate_seconds_shared"] = time.perf_counter() - start
        check_idle()
        start = time.perf_counter()
        dataset = lgb.Dataset(x[:train_rows], label=y[:train_rows], params=params, free_raw_data=True)
        dataset.construct()
        report["dataset_construct_seconds_shared"] = time.perf_counter() - start
        for backend in ("cpu", "hybrid", "resident"):
            model, elapsed = fit(backend, 1)
            report["warmups"].append({"backend": backend, "seconds": elapsed})
            del model
            gc.collect()
        order = ["resident", "hybrid", "cpu", "cpu", "hybrid", "resident"]
        report["run_order"] = order
        for backend in order:
            model, elapsed = fit(backend, rounds)
            start = time.perf_counter()
            pred = model.predict(x[train_rows:], num_threads=threads)
            predict_seconds = time.perf_counter() - start
            if not np.isfinite(pred).all():
                raise RuntimeError(f"{backend}: nonfinite predictions")
            run = {
                "backend": backend,
                "fit_seconds": elapsed,
                "predict_seconds": predict_seconds,
                "auc": float(roc_auc_score(y[train_rows:], pred)),
                "average_precision": float(average_precision_score(y[train_rows:], pred)),
                "log_loss": float(log_loss(y[train_rows:], pred)),
                "model_sha256": digest_model(model),
                "prediction_sha256": digest_predictions(pred),
            }
            previous = next((r for r in report["runs"] if r["backend"] == backend), None)
            if previous and any(run[key] != previous[key] for key in ("model_sha256", "prediction_sha256")):
                raise RuntimeError(f"{backend}: repeated model or prediction changed")
            predictions[backend] = pred
            report["runs"].append(run)
            print(json.dumps(run), flush=True)
            save()
            del model
            gc.collect()
        medians = {
            backend: statistics.median(r["fit_seconds"] for r in report["runs"] if r["backend"] == backend)
            for backend in ("cpu", "hybrid", "resident")
        }
        report["median_fit_seconds"] = medians
        report["hybrid_over_resident_fit_ratio"] = medians["hybrid"] / medians["resident"]
        report["cpu_over_resident_fit_ratio"] = medians["cpu"] / medians["resident"]
        report["prediction_differences"] = {}
        for baseline in ("cpu", "hybrid"):
            delta = np.abs(predictions["resident"] - predictions[baseline])
            report["prediction_differences"][baseline] = {
                "maximum": float(delta.max()),
                "mean": float(delta.mean()),
                "fraction_above_0_001": float(np.mean(delta > 0.001)),
                "fraction_above_0_01": float(np.mean(delta > 0.01)),
            }
        resident = next(r for r in report["runs"] if r["backend"] == "resident")
        for baseline in ("cpu", "hybrid"):
            ref = next(r for r in report["runs"] if r["backend"] == baseline)
            if (
                resident["auc"] < ref["auc"] - 0.001
                or resident["average_precision"] < ref["average_precision"] - 0.005
                or resident["log_loss"] > ref["log_loss"] * 1.02 + 0.0001
            ):
                raise RuntimeError(f"Generated-data quality gate failed against {baseline}")
        report["status"] = "PASS"
    except Exception as error:
        report["status"] = "FAIL"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        os.environ.pop("LGBM_METAL_RESIDENT", None)
        report["peak_process_rss_gib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30
        report["load_average_after"] = list(os.getloadavg())
        save()
    print(f"PASS {args.output}", flush=True)


if __name__ == "__main__":
    main()
