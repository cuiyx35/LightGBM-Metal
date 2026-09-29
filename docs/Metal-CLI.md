# Metal 实验 CLI 与本地看板

**简体中文** · [English](Metal-CLI.en.md)

这是仓库内的开发与验证工具，统一调用现有的构建、生成数据验证和 CPU/Metal 基准脚本。不会读取真实数据文件。所有命令都在仓库根目录执行，使用 Python 3.10 或更新版本。

## 快速使用

~~~bash
python3 tools/metal_experiment.py doctor --json
python3 tools/metal_experiment.py build --validate
python3 tools/metal_experiment.py benchmark
python3 tools/metal_experiment.py report
python3 tools/metal_experiment.py serve
~~~

`serve` 会显示 `http://127.0.0.1:8765/`，只监听本机，每 3 秒读取一次结果。可在另一个终端运行 `build`、`validate` 或 `benchmark`；看板会显示当前命令状态、验证用例和 CPU/Metal 耗时。按 Ctrl+C 停止服务。它显示命令进展和结果，**不提供每棵树的 GPU 利用率或训练内部实时曲线**。

不启动服务也可用 `report` 生成独立 HTML 文件，默认写入 `build-metal/metal_report.html`。要查看已有的公开基准：

~~~bash
python3 tools/metal_experiment.py report \
  --benchmark benchmarks/metal/m5_current_3m_100trees_cpu_metal.json \
  --validation benchmarks/metal/m5_validation.json \
  --output build-metal/public_report.html
~~~

看板只嵌入这些 JSON 中的汇总字段。不要把含私有信息的 JSON 或 HTML 提交到公开仓库。

## 给脚本和 AI 工具调用

| 子命令 | 用途 | 主要输出 |
| --- | --- | --- |
| `doctor --json` | 检查 Apple Silicon、Python、CMake、Clang、OpenMP、验证依赖 | 单个 JSON 对象，退出码 0 表示构建前提齐备 |
| `build [--validate] [--jobs N]` | 编译原生库，可同时运行两组验证 | `build-metal/` 下的库与验证 JSON |
| `validate [--no-resident]` | 运行生成数据的模型级检查 | `build-metal/metal_validation.json` 等 |
| `benchmark` | 运行默认小规模 CPU/Metal 对照 | `build-metal/metal_quick_benchmark.json` |
| `report` | 从已有 JSON 生成静态 HTML | `build-metal/metal_report.html` |
| `serve [--port N]` | 本机自动刷新的看板 | 本机 HTTP 地址 |
| `wheel` | 构建实验性 Apple Silicon Python wheel | `dist/lightgbm-*.whl` |

`build`、`validate`、`benchmark` 和 `wheel` 可用 `--json-out PATH` 写入统一状态 JSON。字段为 `schema_version`、`command`、`status`（`RUNNING`、`PASS`、`FAIL`）、`started_at`、`finished_at`、`message`、`outputs`。运行时同样更新 `build-metal/experiment_status.json`；失败时退出码非零。训练基准的详细配置、计时、预测差仍以基准 JSON 为准，不应从状态文件推断模型质量。

完整规模基准需要显式传 `benchmark --full-scale`，会使用约 410 万行、512 特征的预设。可用 `--train-rows`、`--held-rows`、`--features`、`--dense-features`、`--rounds`、`--threads`、`--repeats` 调整。大型测试可能显著占用统一内存；先确认机器没有其他训练进程。默认小测试无需 `--full-scale`。

## Python wheel：构建一次，安装时免编译

现在的源码方式仍要求每位使用者配置 CMake、C++ 工具链和 OpenMP。仓库新增 `wheel` 命令，将这些编译步骤放在打包机上完成：

~~~bash
brew install cmake libomp
python3 -m pip install build
python3 tools/metal_experiment.py wheel
~~~

打包机必须是 Apple Silicon macOS；构建过程仍需完整工具链。随后在兼容的 Apple Silicon macOS 环境中安装生成的 wheel 时，不再编译 LightGBM：

~~~bash
brew install libomp
python3 -m pip install dist/lightgbm-*.whl
~~~

wheel 保留 `lightgbm` 包名与 4.7.0 版本号，会替换当前 Python 环境里同名的官方 LightGBM；请先建独立虚拟环境。它依赖系统提供的 `libomp`，尚未做到“下载后零依赖”。不要将此 wheel 上传到 PyPI，也不要当作官方发行版。GitHub Actions 的 Metal macOS 工作流会在干净的虚拟环境中安装 wheel 并做 Metal 小型训练，成功后把它作为工作流构件提供；下载时需核对工作流提交与运行结果。
