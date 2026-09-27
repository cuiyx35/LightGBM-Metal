"""Generated-data checks for the opt-in GPU-resident tree learner.

Checks CPU-reference split gains and exact row partitions in native code,
training-score consistency, saved-model prediction, and held-out metrics.
Timings include diagnostics and must not be used as performance results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import log_loss, mean_squared_error, roc_auc_score

import lightgbm as lgb


class CaptureLogger:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def info(self, message: str) -> None:
        self.messages.append(message)

    def warning(self, message: str) -> None:
        self.messages.append(message)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("metal_resident_validation.json"))
    parser.add_argument("--case", action="append", dest="cases", help="Run only this named case (repeatable)")
    args = parser.parse_args()
    report: dict = {"scope": "generated_data_correctness_with_native_diagnostics", "cases": []}
    logger = CaptureLogger()
    lgb.register_logger(logger)
    previous = {name: os.environ.get(name) for name in ("LGBM_METAL_RESIDENT", "LGBM_METAL_RESIDENT_VERIFY")}

    def run(
        name,
        x,
        y,
        objective="binary",
        extra=None,
        dataset_params=None,
        weights=None,
        fallback=False,
        reset_parameters=None,
        replace_training_data=False,
        resume=False,
    ):
        if args.cases and name not in args.cases:
            return
        boundary = int(len(y) * 0.75)
        train = x.iloc[:boundary] if isinstance(x, pd.DataFrame) else x[:boundary]
        held = x.iloc[boundary:] if isinstance(x, pd.DataFrame) else x[boundary:]
        settings = {"max_bin": 63, "feature_pre_filter": False, **(dataset_params or {})}
        params = {
            "objective": objective,
            "num_threads": 2,
            "num_leaves": 15,
            "seed": 471,
            "force_col_wise": True,
            "verbosity": 1,
            "learning_rate": 0.1,
            **settings,
            **(extra or {}),
        }
        dataset = lgb.Dataset(
            train,
            label=y[:boundary],
            weight=None if weights is None else weights[:boundary],
            free_raw_data=False,
            params=settings,
        )
        predictions, metrics, rows = {}, {}, []
        for backend in ("cpu", "hybrid", "resident"):
            print(f"CASE {name} BACKEND {backend}", flush=True)
            os.environ["LGBM_METAL_RESIDENT"] = "1" if backend == "resident" else "0"
            os.environ["LGBM_METAL_RESIDENT_VERIFY"] = "1"
            logger.messages.clear()
            backend_params = {**params, "device_type": "cpu" if backend == "cpu" else "metal"}
            initial = None
            if resume:
                initial = lgb.train(backend_params, dataset, num_boost_round=5)
                initial = lgb.Booster(model_str=initial.model_to_string())
            model = lgb.train(
                backend_params,
                dataset,
                init_model=initial,
                num_boost_round=10 if resume else 15,
                keep_training_booster=True,
                callbacks=[] if reset_parameters is None else [lgb.reset_parameter(**reset_parameters)],
            )
            score_train = train
            if replace_training_data:
                half = boundary // 2
                score_train = train.iloc[:half] if isinstance(train, pd.DataFrame) else train[:half]
                replacement = lgb.Dataset(
                    score_train, label=y[:half], reference=dataset, params=settings, free_raw_data=False
                )
                model.update(train_set=replacement)
            cached = model._Booster__inner_predict(data_idx=0).copy()
            fresh = model.predict(score_train)
            score_error = float(np.max(np.abs(cached - fresh)))
            if score_error > 1e-10:
                raise AssertionError(f"{name}/{backend}: cached scores disagree with prediction: {score_error}")
            prediction = model.predict(held)
            reloaded = lgb.Booster(model_str=model.model_to_string())
            reload_error = float(np.max(np.abs(prediction - reloaded.predict(held))))
            if reload_error > 1e-12 or not np.isfinite(prediction).all():
                raise AssertionError(f"{name}/{backend}: invalid prediction or model round trip")
            if backend == "resident":
                marker = "resident tree fallback:" if fallback else "resident tree:"
                if not any(marker in message for message in logger.messages):
                    raise AssertionError(f"{name}: expected execution path was not logged: {marker}")
            if objective == "binary":
                metric = {
                    "auc": float(roc_auc_score(y[boundary:], prediction)),
                    "loss": float(log_loss(y[boundary:], prediction)),
                }
            elif objective == "multiclass":
                metric = {"loss": float(log_loss(y[boundary:], prediction))}
            else:
                metric = {"mse": float(mean_squared_error(y[boundary:], prediction))}
            metrics[backend] = metric
            predictions[backend] = prediction
            rows.append(
                {
                    "backend": backend,
                    "metric": metric,
                    "cached_score_max_error": score_error,
                    "reload_max_error": reload_error,
                    "model_sha256": hashlib.sha256(model.model_to_string().encode()).hexdigest(),
                    "native_verification_trees": sum("resident verification:" in m for m in logger.messages),
                }
            )
        # These are fixed engineering gates for generated tasks, not business-model acceptance criteria.
        for baseline in ("cpu", "hybrid"):
            if "auc" in metrics[baseline] and metrics["resident"]["auc"] < metrics[baseline]["auc"] - 0.005:
                raise AssertionError(f"{name}: held-out AUC gate failed: {metrics}")
            loss = "mse" if "mse" in metrics[baseline] else "loss"
            if metrics["resident"][loss] > metrics[baseline][loss] * 1.05 + 0.002:
                raise AssertionError(f"{name}: held-out loss gate failed: {metrics}")
        if fallback and not np.array_equal(predictions["resident"], predictions["hybrid"]):
            raise AssertionError(f"{name}: fallback changed hybrid predictions")
        result = {
            "name": name,
            "status": "PASS",
            "expected_fallback": fallback,
            "runs": rows,
            "cpu_resident_max_prediction_difference": float(
                np.max(np.abs(predictions["cpu"] - predictions["resident"]))
            ),
        }
        report["cases"].append(result)
        print(json.dumps(result), flush=True)

    try:
        for seed in (13, 99, 471):
            rng = np.random.default_rng(seed)
            x = rng.normal(size=(8000, 12)).astype(np.float32)
            y = (x[:, 0] + x[:, 1] * x[:, 2] + rng.normal(scale=0.25, size=len(x)) > 0).astype(np.int8)
            run(f"numeric_{seed}", x, y)
            mixed = x.copy()
            mixed[rng.random(mixed.shape) < 0.15] = np.nan
            mixed[:, 6:] = np.nan_to_num(mixed[:, 6:]) * (rng.random((len(x), 6)) < 0.05)
            frame = pd.DataFrame(mixed, columns=[f"f{i}" for i in range(12)])
            cats = rng.integers(0, 30, size=len(x))
            frame["category"] = pd.Categorical(cats)
            ym = (
                np.nan_to_num(mixed[:, 0])
                + 0.8 * np.nan_to_num(mixed[:, 1])
                + 0.9 * (cats % 3 == 0)
                + rng.normal(scale=0.3, size=len(x))
                > 0.3
            ).astype(np.int8)
            run(
                f"mixed_bagging_{seed}",
                frame,
                ym,
                extra={"feature_fraction": 0.8, "bagging_fraction": 0.75, "bagging_freq": 1},
            )
        run(
            "regularization_depth",
            frame,
            ym,
            extra={
                "lambda_l1": 0.3,
                "lambda_l2": 2.0,
                "path_smooth": 3.0,
                "max_delta_step": 0.8,
                "max_depth": 4,
                "cat_l2": 4.0,
                "cat_smooth": 5.0,
                "min_data_per_group": 30,
                "max_cat_threshold": 9,
            },
        )
        run("zero_missing", mixed, ym, dataset_params={"zero_as_missing": True})
        run("missing_disabled", mixed, ym, dataset_params={"use_missing": False})
        run("variable_weights", frame, ym, weights=rng.uniform(0.01, 3.0, len(ym)))
        frame["category"] = pd.Categorical(cats % 3)
        run("onehot_categories", frame, ym)
        sparse_x = rng.normal(size=(8000, 96)).astype(np.float32) * (rng.random((8000, 96)) < 0.02)
        sparse_y = (sparse_x[:, :16].sum(axis=1) + rng.normal(scale=0.1, size=8000) > 0).astype(np.int8)
        run("sparse_multi_value", sparse.csr_matrix(sparse_x), sparse_y)
        reg_y = 1.2 * x[:, 0] - 0.5 * x[:, 1] + rng.normal(scale=0.2, size=len(x))
        run("constant_hessian", x, reg_y, objective="regression")
        run("objective_leaf_renewal", x, reg_y, objective="regression_l1")
        run("multiclass", x, np.digitize(reg_y, [-0.5, 0.5]), objective="multiclass", extra={"num_class": 3})
        run("stump", x, y, extra={"min_gain_to_split": 1e8})
        run("wide_bins_fallback", x, y, dataset_params={"max_bin": 511}, fallback=True)
        run("monotone_fallback", x, y, extra={"monotone_constraints": [1] + [0] * 11}, fallback=True)
        run("node_sampling_fallback", x, y, extra={"feature_fraction_bynode": 0.8}, fallback=True)
        run("extra_trees_fallback", x, y, extra={"extra_trees": True}, fallback=True)
        run("resize_leaf_pool", x, y, reset_parameters={"num_leaves": [7] * 5 + [31] * 5 + [15] * 5})
        run(
            "switch_fallback_mid_fit",
            x,
            y,
            reset_parameters={"feature_fraction_bynode": [1.0] * 5 + [0.8] * 5 + [1.0] * 5},
        )
        run(
            "goss",
            x,
            y,
            extra={"data_sample_strategy": "goss", "learning_rate": 0.4, "top_rate": 0.2, "other_rate": 0.1},
        )
        run("replace_training_dataset", x, y, replace_training_data=True)
        run("resume_serialized_model", frame, ym, resume=True)
        run("maximum_supported_bins", x, y, dataset_params={"max_bin": 256, "min_data_in_bin": 1})
        run("zero_weight_rows", x, y, weights=(rng.random(len(y)) > 0.3).astype(np.float32))
        if args.cases and set(args.cases) != {case["name"] for case in report["cases"]}:
            raise ValueError("Unknown --case name")
        report["status"] = "PASS"
    except Exception as error:
        report["status"] = "FAIL"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
