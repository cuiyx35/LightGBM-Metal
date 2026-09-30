#!/usr/bin/env python3
"""Local CLI and dashboard for the experimental Apple Silicon Metal backend.

This tool never reads a dataset. Training commands delegate to the existing
generated-data validation and benchmark scripts. The dashboard reads only the
JSON paths selected by the caller and listens on loopback only.
"""

from __future__ import annotations

import argparse
import html
import importlib.util
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build-metal"
STATUS = BUILD / "experiment_status.json"
DEFAULT_BENCHMARK = BUILD / "metal_quick_benchmark.json"
DEFAULT_VALIDATION = BUILD / "metal_validation.json"
DEFAULT_RESIDENT_VALIDATION = BUILD / "metal_resident_validation.json"
SCHEMA_VERSION = 1


def now() -> str:
    """Return a UTC timestamp for experiment status records."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path: Path, data: dict) -> None:
    """Atomically replace a status or result JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(data, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    temporary.replace(path)


def read_json(path: Path | None) -> tuple[dict | None, str | None]:
    """Read an optional report and return any validation error."""
    if path is None:
        return None, None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise TypeError("top-level JSON value must be an object")
        data["_source_name"] = path.name
        return data, None
    except FileNotFoundError:
        return None, None
    except (OSError, TypeError, ValueError) as error:
        return None, f"{path.name}: {error}"


def text(value: object) -> str:
    """Escape a value for inclusion in dashboard HTML."""
    return html.escape(str(value), quote=True)


def number(value: object, digits: int = 3) -> str:
    """Format a finite numeric value or a missing-value marker."""
    if not isinstance(value, (float, int)) or isinstance(value, bool) or not math.isfinite(value):
        return "—"
    return f"{value:,.{digits}f}"


def check_environment() -> dict:
    """Inspect local prerequisites without building or training."""
    apple = platform.system() == "Darwin" and platform.machine() == "arm64"
    cmake = shutil.which(os.environ.get("CMAKE_BIN", "cmake"))
    clang = shutil.which("clang")
    python_ok = sys.version_info >= (3, 10)
    prefix = os.environ.get("OPENMP_PREFIX")
    if not prefix and shutil.which("brew"):
        result = subprocess.run(["brew", "--prefix", "libomp"], text=True, capture_output=True, check=False)
        if result.returncode == 0:
            prefix = result.stdout.strip()
    omp = bool(prefix and (Path(prefix) / "include/omp.h").is_file() and (Path(prefix) / "lib/libomp.dylib").is_file())
    packages = {
        name: importlib.util.find_spec(name) is not None for name in ("numpy", "scipy", "pandas", "sklearn", "narwhals")
    }
    checks = {
        "apple_silicon_macos": apple,
        "python_3_10_or_newer": python_ok,
        "cmake": bool(cmake),
        "clang": bool(clang),
        "openmp": omp,
        "validation_packages": all(packages.values()),
        "built_library": (ROOT / "lib_lightgbm.dylib").is_file(),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "command": "doctor",
        "status": "PASS"
        if all(
            checks[name]
            for name in (
                "apple_silicon_macos",
                "python_3_10_or_newer",
                "cmake",
                "clang",
                "openmp",
            )
        )
        else "INCOMPLETE",
        "checks": checks,
        "python_packages": packages,
        "python": sys.executable,
        "cmake": cmake,
        "openmp_prefix": prefix,
        "note": "Validation packages are needed for tests; a downloaded native wheel avoids local compilation but still needs libomp at runtime.",
    }


def environment() -> dict[str, str]:
    """Prepare the checkout's Python environment for child commands."""
    env = os.environ.copy()
    env["PYTHON_BIN"] = sys.executable
    env["PYTHONPATH"] = str(ROOT / "python-package") + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return env


def run_commands(
    command_name: str,
    commands: list[list[str]],
    outputs: list[Path],
    json_out: Path | None,
) -> int:
    """Run commands sequentially and record their completion status."""
    state = {
        "schema_version": SCHEMA_VERSION,
        "command": command_name,
        "status": "RUNNING",
        "started_at": now(),
        "message": "Starting",
        "pid": os.getpid(),
        "outputs": [str(path) for path in outputs],
    }
    write_json(STATUS, state)
    try:
        for command in commands:
            state["message"] = "Running " + Path(command[1] if len(command) > 1 else command[0]).name
            write_json(STATUS, state)
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=environment(),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                stripped = line.strip()
                if stripped.startswith(
                    (
                        "DATA_READY",
                        "DATASET_READY",
                        "RUN ",
                        "CASE ",
                        "PASS ",
                        "[ERROR]",
                        "CMake Error",
                    )
                ):
                    state["message"] = stripped[:500]
                    write_json(STATUS, state)
            code = process.wait()
            if code:
                raise RuntimeError(f"{command_name} exited with code {code}")
        state["status"] = "PASS"
        state["message"] = "Completed"
        code = 0
    except (OSError, RuntimeError) as error:
        state["status"] = "FAIL"
        state["message"] = str(error)
        code = 1
        print(str(error), file=sys.stderr)
    state["finished_at"] = now()
    write_json(STATUS, state)
    if json_out is not None:
        write_json(json_out, state)
    return code


def panel(title: str, body: str) -> str:
    """Wrap dashboard content in a titled HTML section."""
    return f'<section class="panel"><h2>{text(title)}</h2>{body}</section>'


def case_panel(title: str, report: dict | None) -> str:
    """Render the individual cases of a validation report."""
    if report is None:
        return panel(title, '<p class="muted">尚无结果</p>')
    cases = report.get("cases", [])
    rows = []
    if isinstance(cases, list):
        for case in cases:
            if isinstance(case, dict):
                name = case.get("name", case.get("case", "?"))
                rows.append(f"<tr><td>{text(name)}</td><td>{text(case.get('status', '?'))}</td></tr>")
    body = f'<p class="badge">{text(report.get("status", "UNKNOWN"))}</p>'
    if report.get("_source_name"):
        body += f'<p class="muted">结果文件：{text(report["_source_name"])}</p>'
    if rows:
        body += "<table><thead><tr><th>测试</th><th>结果</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
    if "error" in report:
        body += f'<p class="error">{text(report["error"])}</p>'
    return panel(title, body)


def benchmark_panel(report: dict | None) -> str:
    """Render benchmark summaries without exposing raw training data."""
    if report is None:
        return panel(
            "CPU / Metal 基准",
            '<p class="muted">尚无结果；可用 benchmark 命令运行小规模合成测试。</p>',
        )
    median = report.get("median", {})
    cpu_data = median.get("cpu") if isinstance(median, dict) else None
    metal_data = median.get("metal") if isinstance(median, dict) else None
    cpu = cpu_data.get("fit_seconds") if isinstance(cpu_data, dict) else None
    metal = metal_data.get("fit_seconds") if isinstance(metal_data, dict) else None
    valid = all(
        isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v >= 0 for v in (cpu, metal)
    )
    cards = (
        f'<div class="cards"><div><small>CPU 训练中位数</small><strong>{number(cpu)} s</strong></div>'
        f"<div><small>Metal 训练中位数</small><strong>{number(metal)} s</strong></div>"
        f"<div><small>CPU / Metal</small><strong>{number(cpu / metal, 2) if valid and metal > 0 else '—'}×</strong></div></div>"
    )
    bars = ""
    if valid and max(cpu, metal) > 0:
        for label, value, css in (("CPU", cpu, "cpu"), ("Metal", metal, "metal")):
            width = 100 * value / max(cpu, metal)
            bars += f'<div class="barrow"><span>{label}</span><div class="track"><div class="bar {css}" style="width:{width:.1f}%"></div></div><b>{number(value)} s</b></div>'
    runs = report.get("runs", [])
    rows = []
    if isinstance(runs, list):
        for run in runs:
            if isinstance(run, dict):
                rows.append(
                    "<tr>"
                    + "".join(f"<td>{text(run.get(key, '—'))}</td>" for key in ("index", "backend"))
                    + f"<td>{number(run.get('fit_seconds'))}</td><td>{number(run.get('average_precision'), 5)}</td></tr>"
                )
    config = report.get("configuration", {})
    config_line = ""
    if isinstance(config, dict):
        config_line = " · ".join(
            f"{text(k)}: {text(config[k])}"
            for k in ("train_rows", "held_rows", "features", "rounds", "threads")
            if k in config
        )
    difference = report.get("metal_vs_cpu_prediction", {})
    diff_line = ""
    if isinstance(difference, dict):
        diff_line = f"预测最大绝对差：{number(difference.get('maximum_absolute_difference'), 6)}；超过 0.001 的行数：{text(difference.get('count_above_0_001', '—'))}"
    body = f'<p class="badge">{text(report.get("status", "UNKNOWN"))}</p><p class="muted">{config_line}</p>{cards}{bars}<p>{diff_line}</p>'
    if report.get("_source_name"):
        body += f'<p class="muted">结果文件：{text(report["_source_name"])}</p>'
    if rows:
        body += (
            "<table><thead><tr><th>顺序</th><th>后端</th><th>训练秒数</th><th>AP</th></tr></thead><tbody>"
            + "".join(rows)
            + "</tbody></table>"
        )
    body += '<p class="muted">计时与质量仅对应本次生成数据；不代表业务模型质量或其他 M 芯片速度。</p>'
    return panel("CPU / Metal 基准", body)


def dashboard(
    benchmark: dict | None,
    validation: dict | None,
    resident: dict | None,
    state: dict | None,
    errors: list[str],
    refresh: bool,
) -> str:
    """Render a complete dashboard from selected summary reports."""
    status = "尚未运行"
    message = ""
    if state:
        status = str(state.get("status", "UNKNOWN"))
        message = str(state.get("message", ""))
        if status == "RUNNING" and isinstance(state.get("pid"), int):
            try:
                os.kill(state["pid"], 0)
            except ProcessLookupError:
                status = "INTERRUPTED"
                message = "命令进程已结束；状态文件未写入最终结果。"
            except PermissionError:
                pass
    warning = "".join(f'<p class="error">{text(error)}</p>' for error in errors)
    live = '<meta http-equiv="refresh" content="3">' if refresh else ""
    body = (
        panel("运行状态", f'<p class="badge">{text(status)}</p><p>{text(message)}</p>')
        + benchmark_panel(benchmark)
        + case_panel("默认协同路径验证", validation)
        + case_panel("可选常驻建树验证", resident)
    )
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">{live}
<title>LightGBM Metal 实验看板</title>
<style>
:root{{color-scheme:light dark;font-family:system-ui,-apple-system,sans-serif;line-height:1.5}}
body{{max-width:1050px;margin:0 auto;padding:2rem;background:#f4f7fb;color:#172436}}
h1{{font-size:1.8rem;margin-bottom:.2rem}}h2{{font-size:1.12rem;margin-top:0}}
.muted{{color:#61748a}}.error{{color:#b22233;overflow-wrap:anywhere}}
.panel{{background:white;border:1px solid #dce5ef;border-radius:14px;padding:1.25rem;margin:1.1rem 0;box-shadow:0 2px 12px #1724360b}}
.badge{{display:inline-block;background:#e8f2fc;color:#155788;padding:.25rem .75rem;border-radius:99px;font-weight:700}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:.75rem;margin:1rem 0}}
.cards>div{{background:#f2f6fb;border-radius:10px;padding:.8rem}}small,strong{{display:block}}strong{{font-size:1.55rem}}
.barrow{{display:flex;align-items:center;gap:.7rem;margin:.65rem 0}}.barrow span{{width:52px}}.barrow b{{width:95px;text-align:right}}
.track{{height:19px;flex:1;background:#e8edf3;border-radius:99px;overflow:hidden}}.bar{{height:100%;border-radius:99px}}.cpu{{background:#6a8caa}}.metal{{background:#0f9d91}}
table{{border-collapse:collapse;width:100%;margin-top:1rem}}th,td{{padding:.55rem;text-align:left;border-bottom:1px solid #e1e8f0}}th{{font-size:.82rem;color:#61748a}}
@media(prefers-color-scheme:dark){{body{{background:#0f1722;color:#e7edf5}}.panel{{background:#182331;border-color:#304154}}.muted,th{{color:#a8b5c7}}.cards>div{{background:#233143}}.track{{background:#344257}}.badge{{background:#24445a;color:#b8ecff}}td,th{{border-color:#304154}}}}
</style></head><body><header><h1>LightGBM Metal 实验看板</h1><p class="muted">本地合成数据验证与基准 · CPU＋GPU 协同路径</p></header>{warning}{body}
<footer class="muted">此页面只展示指定 JSON 的汇总，不读取原始训练数据。{("本地服务每 3 秒刷新。" if refresh else "静态报告；不会自动刷新。")}</footer></body></html>"""


def load_dashboard(
    args: argparse.Namespace,
) -> tuple[dict | None, dict | None, dict | None, dict | None, list[str]]:
    """Read selected dashboard reports and collect input errors."""
    paths = (
        args.benchmark,
        args.validation,
        args.resident_validation,
        args.status_file,
    )
    read = [read_json(path) for path in paths]
    return (*(item[0] for item in read), [item[1] for item in read if item[1]])


def add_dashboard_paths(parser: argparse.ArgumentParser) -> None:
    """Register the report paths shared by report and serve commands."""
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--validation", type=Path, default=DEFAULT_VALIDATION)
    parser.add_argument("--resident-validation", type=Path, default=DEFAULT_RESIDENT_VALIDATION)
    parser.add_argument("--status-file", type=Path, default=STATUS)


def _doctor(args: argparse.Namespace) -> int:
    """Print prerequisite checks and return their status."""
    result = check_environment()
    if args.json:
        print(json.dumps(result, ensure_ascii=False))
    else:
        for name, passed in result["checks"].items():
            print(f"{'OK' if passed else 'MISSING'} {name}")
        print("OpenMP:", result["openmp_prefix"] or "not found")
    return 0 if result["status"] == "PASS" else 1


def _report(args: argparse.Namespace) -> int:
    """Render a selected summary dashboard as a standalone file."""
    benchmark_data, validation_data, resident_data, state, errors = load_dashboard(args)
    if errors:
        print("; ".join(errors), file=sys.stderr)
        return 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        dashboard(benchmark_data, validation_data, resident_data, state, errors, False),
        encoding="utf-8",
    )
    print(args.output.resolve())
    return 0


def _serve(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    """Serve selected summary reports on loopback."""
    if not 0 <= args.port <= 65535:
        parser.error("--port must be between 0 and 65535")

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path not in ("/", "/index.html"):
                self.send_error(404)
                return
            data = dashboard(*load_dashboard(args), True).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    with ThreadingHTTPServer(("127.0.0.1", args.port), Handler) as server:
        print(f"http://127.0.0.1:{server.server_port}/", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
    return 0


def _wheel(args: argparse.Namespace) -> int:
    """Build the experimental Metal wheel and record its status."""
    return run_commands(
        "wheel",
        [["bash", str(ROOT / "tools/build-metal-wheel.sh")]],
        [ROOT / "dist"],
        args.json_out,
    )


def main() -> int:
    """Parse and dispatch one experiment CLI subcommand."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    doctor = sub.add_parser("doctor", help="Check build and validation prerequisites")
    doctor.add_argument("--json", action="store_true", help="Print one machine-readable JSON object")
    build = sub.add_parser("build", help="Compile the Metal library in this checkout")
    build.add_argument("--jobs", type=int, default=2)
    build.add_argument("--validate", action="store_true")
    build.add_argument("--json-out", type=Path)
    validate = sub.add_parser("validate", help="Run generated-data model checks")
    validate.add_argument("--no-resident", action="store_true")
    validate.add_argument("--json-out", type=Path)
    benchmark = sub.add_parser("benchmark", help="Run the synthetic CPU/Metal comparison")
    benchmark.add_argument("--full-scale", action="store_true", help="Explicitly allow the large preset")
    benchmark.add_argument("--output", type=Path, default=DEFAULT_BENCHMARK)
    benchmark.add_argument("--json-out", type=Path)
    for name in (
        "train-rows",
        "held-rows",
        "features",
        "dense-features",
        "rounds",
        "threads",
        "repeats",
    ):
        benchmark.add_argument("--" + name, type=int)
    report = sub.add_parser("report", help="Write a standalone HTML dashboard")
    add_dashboard_paths(report)
    report.add_argument("--output", type=Path, default=BUILD / "metal_report.html")
    serve = sub.add_parser("serve", help="Serve an auto-refreshing dashboard on 127.0.0.1")
    add_dashboard_paths(serve)
    serve.add_argument("--port", type=int, default=8765)
    wheel = sub.add_parser("wheel", help="Build an experimental Apple Silicon Python wheel")
    wheel.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    handlers = {"doctor": _doctor, "report": _report, "serve": lambda args: _serve(args, parser), "wheel": _wheel}
    if args.command in handlers:
        return handlers[args.command](args)
    if args.command == "build":
        if args.jobs < 1:
            parser.error("--jobs must be positive")
        command = [
            "bash",
            str(ROOT / "tools/build-metal-macos.sh"),
            "--jobs",
            str(args.jobs),
        ]
        if args.validate:
            command.append("--validate")
        outputs = [ROOT / "lib_lightgbm.dylib"]
        if args.validate:
            outputs += [DEFAULT_VALIDATION, DEFAULT_RESIDENT_VALIDATION]
        return run_commands("build", [command], outputs, args.json_out)
    if args.command == "validate":
        scripts = [("metal_validate.py", DEFAULT_VALIDATION)]
        if not args.no_resident:
            scripts.append(("metal_resident_validate.py", DEFAULT_RESIDENT_VALIDATION))
        commands = [
            [
                sys.executable,
                str(ROOT / "examples/python-guide" / name),
                "--output",
                str(output),
            ]
            for name, output in scripts
        ]
        return run_commands("validate", commands, [output for _, output in scripts], args.json_out)
    if args.command == "benchmark":
        command = [
            sys.executable,
            str(ROOT / "examples/python-guide/metal_synthetic_benchmark.py"),
        ]
        if args.full_scale:
            command.append("--full-scale")
        for name in (
            "train_rows",
            "held_rows",
            "features",
            "dense_features",
            "rounds",
            "threads",
            "repeats",
        ):
            value = getattr(args, name)
            if value is not None:
                command += ["--" + name.replace("_", "-"), str(value)]
        command += ["--output", str(args.output)]
        return run_commands("benchmark", [command], [args.output], args.json_out)
    return 2


if __name__ == "__main__":
    sys.exit(main())
