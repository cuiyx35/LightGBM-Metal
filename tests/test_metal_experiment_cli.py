"""Dependency-free checks for the local Metal experiment CLI and dashboard."""

import importlib.util
import json
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "metal_experiment.py"
SPEC = importlib.util.spec_from_file_location("metal_experiment", SCRIPT)
assert SPEC and SPEC.loader
metal_experiment = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(metal_experiment)


def test_dashboard_renders_real_benchmark_and_validation():
    root = SCRIPT.parents[1]
    benchmark = json.loads(
        (root / "benchmarks/metal/m5_current_3m_100trees_cpu_metal.json").read_text()
    )
    validation = json.loads((root / "benchmarks/metal/m5_validation.json").read_text())
    page = metal_experiment.dashboard(
        benchmark, validation, None, {"status": "PASS"}, [], False
    )
    assert "CPU / Metal 基准" in page
    assert "1.67×" in page
    assert "默认协同路径验证" in page
    assert 'meta http-equiv="refresh"' not in page


def test_dashboard_escapes_untrusted_json():
    page = metal_experiment.dashboard(
        {"status": "PASS", "configuration": {"features": "<script>alert(1)</script>"}},
        {"status": "FAIL", "error": "<img src=x onerror=alert(1)>"},
        None,
        {"status": "RUNNING", "message": "<svg/onload=alert(1)>"},
        [],
        True,
    )
    assert "<script>alert(1)</script>" not in page
    assert "<img src=x onerror=alert(1)>" not in page
    assert "<svg/onload=alert(1)>" not in page
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in page
    assert 'meta http-equiv="refresh"' in page


def test_dashboard_handles_incomplete_benchmark_shape():
    page = metal_experiment.dashboard(
        {"median": {"cpu": "unexpected"}}, None, None, None, [], False
    )
    assert "CPU / Metal 基准" in page
    assert "—" in page


def test_command_writes_machine_readable_completion(tmp_path, monkeypatch):
    monkeypatch.setattr(metal_experiment, "STATUS", tmp_path / "live.json")
    result = tmp_path / "result.json"
    exit_code = metal_experiment.run_commands(
        "smoke", [[sys.executable, "-c", 'print("DATA_READY synthetic")']], [], result
    )
    assert exit_code == 0
    assert json.loads(result.read_text())["status"] == "PASS"
    assert json.loads((tmp_path / "live.json").read_text())["message"] == "Completed"
