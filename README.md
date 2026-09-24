# LightGBM Metal：Apple 芯片实验版

**简体中文** · [English](README.en.md)

> [!IMPORTANT]
> 这是基于 LightGBM 4.7.0 的独立实验性分支，并非 LightGBM 官方版本。Metal 后端目前只在一台 Apple M5 上完成性能测试；请先验证自己的数据和模型质量，再考虑实际使用。

本项目为 Apple Silicon 增加 <code>device_type="metal"</code>。训练时，GPU 计算符合条件的特征组直方图，CPU 同时计算其他特征组的直方图，并负责寻找分裂点和构建树。预测和模型保存仍使用 LightGBM 常规路径。它是 **CPU 与 GPU 协同训练**，并非全程只用 GPU。

## 当前能力与边界

- Metal 路径处理单特征、非 multi-value、最多 256 个 bin 的特征组；其余组由 CPU 处理。
- GPU 直方图使用定点量化。分裂点接近时，生成的树和预测可能与纯 CPU 训练不同。不能只凭预测差小或合成数据指标相近，推断业务模型质量相同。
- 构建需要 Apple Silicon macOS、Metal GPU、CMake、C++ 工具链，以及可用的 OpenMP 运行时。Python 验证脚本还需要 NumPy、pandas 和 scikit-learn。
- 尚未在其他 M 系列芯片上复核性能；加速比会随数据分布、芯片、线程数、温度及后台负载变化。

详细架构、限制和参数见[实验使用说明](docs/Metal-Experimental.md)；想继续开发请读[开发指南](docs/Metal-Development.md)。

## 构建和快速验证

在本仓库根目录执行；按本机工具链配置 OpenMP：

~~~bash
cmake -S . -B build-metal -DCMAKE_BUILD_TYPE=Release -DUSE_METAL=ON
cmake --build build-metal -j2
ln -sfn ../lib_lightgbm.dylib python-package/lib_lightgbm.dylib
export PYTHONPATH="$PWD/python-package"
python examples/python-guide/metal_validate.py --output metal_validation.json
python examples/python-guide/metal_synthetic_benchmark.py --output metal_quick_benchmark.json
~~~

确认 Python 导入的是此仓库的 <code>lightgbm</code> 和启用 Metal 的 <code>lib_lightgbm.dylib</code>。验证脚本用生成数据检查五类模型级情形，结果写入 JSON；失败时退出码非零。基准脚本默认运行小规模冒烟测试。使用 Metal 训练时设置 <code>device_type="metal"</code>；返回 CPU 路径设置 <code>device_type="cpu"</code>。

## 公开测试结果

一台配备 32 GiB 统一内存的 Apple M5，在生成数据的 330 万训练行、80 万留出行、512 个特征、500 棵树、6 个 CPU 线程的测试中，CPU/Metal **训练时间中位数之比为 1.55 倍**。训练加预测的中位数之比为 1.53 倍；加入双方共用的一次性数据生成和 Dataset 构建时间后，估算整体比值为 1.45 倍。这是两次运行每种后端的结果，不能代表其他机器或真实业务数据。完整配置、原始 JSON、模型差异及限制见[公开基准报告](benchmarks/metal/README.md)。

运行同量级合成测试需较多统一内存，并且必须显式指定 <code>--full-scale</code>：

~~~bash
python examples/python-guide/metal_synthetic_benchmark.py \
  --full-scale --output synthetic_metal_result.json
~~~

脚本只生成本地合成数据，不读取外部数据文件；公开报告不包含逐行预测。

## Star History

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/star-history-dark.svg" />
  <img alt="LightGBM-Metal 的 Star History" src="assets/star-history.svg" />
</picture>

图由[本仓库的 GitHub Actions](.github/workflows/star-history.yml)每周用临时仓库令牌从 GitHub 数据生成；不需要向第三方图表服务提供个人令牌。新仓库在出现第一颗星前会显示 0 的平线。

## 文档与二次开发

| 内容 | 中文 | English |
| --- | --- | --- |
| 项目入口 | 本页 | [README.en.md](README.en.md) |
| Metal 使用与验证 | [实验使用说明](docs/Metal-Experimental.md) | [Experimental Metal notes](docs/Metal-Experimental.en.md) |
| 架构与扩展流程 | [开发指南](docs/Metal-Development.md) | [Development guide](docs/Metal-Development.en.md) |
| 测试结果与限制 | [公开基准报告](benchmarks/metal/README.md) | [Benchmark report](benchmarks/metal/README.en.md) |
| AI 编码约束 | [AGENTS.md](AGENTS.md) | [AGENTS.en.md](AGENTS.en.md) |

上表覆盖本实验分支的专属说明；通用 LightGBM 上游文档仍保持原文。

验证和基准脚本提供命令行参数、退出码与 JSON 输出，适合在 CI 或 AI 辅助开发流程中调用。更改 Metal 后端时，保留 <code>USE_METAL=OFF</code> 下的上游行为，并使用生成数据检查模型结构、预测、性能和回退路径。

## 来源与许可

本仓库从 LightGBM 4.7.0 的[官方提交 <code>8f7036f</code>](https://github.com/lightgbm-org/LightGBM/commit/8f7036f03627054d5a54a6f965b13f4b9ff2cb63)导入源码快照；更早的历史保留在[官方 LightGBM 仓库](https://github.com/lightgbm-org/LightGBM)。项目保留上游 [MIT 许可证](LICENSE)和第三方许可声明。上游文档、安装说明与通用 LightGBM 功能请以[官方文档](https://lightgbm.readthedocs.io/)为准。
