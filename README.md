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

在本仓库根目录执行。先安装 CMake 和 OpenMP 运行时，并为验证准备 NumPy、SciPy、pandas、scikit-learn、narwhals。使用 Homebrew 时，构建脚本会自动找到 <code>libomp</code>：

~~~bash
brew install cmake libomp
bash tools/build-metal-macos.sh --validate
~~~

脚本在 <code>build-metal/</code> 中构建，将原生库链接到本仓库的 Python 包；<code>--validate</code> 用生成数据检查五类模型级情形，将 JSON 写入 <code>build-metal/metal_validation.json</code>，失败时退出码非零。若 OpenMP 安装在其他位置，可设置 <code>OPENMP_PREFIX</code>；使用不同 Python 可设置 <code>PYTHON_BIN</code>。<code>bash tools/build-metal-macos.sh --help</code> 列出全部参数。GitHub Actions 的 [Metal macOS 检查](.github/workflows/metal_macos.yml)使用同一脚本在 Apple Silicon 托管运行器上构建并验证；它不运行性能基准。

如需单独做小规模性能冒烟测试，先构建，然后执行：

~~~bash
PYTHONPATH="$PWD/python-package" python3 examples/python-guide/metal_synthetic_benchmark.py --output metal_quick_benchmark.json
~~~

使用 Metal 训练时设置 <code>device_type="metal"</code>；返回 CPU 路径设置 <code>device_type="cpu"</code>。
连续比较多个训练配置时，可先构建一次 <code>lightgbm.Dataset</code> 并重复传给 <code>lightgbm.train()</code>，省去重复分箱。仅对训练行、特征顺序、类别定义和分箱参数相同的实验复用；改变 <code>max_bin</code> 或交叉验证切分时需重建。示例见[实验使用说明](docs/Metal-Experimental.md#重复实验复用-dataset)。

## 公开测试结果

一台配备 32 GiB 统一内存的 Apple M5，在生成数据的 330 万训练行、80 万留出行、512 个特征、500 棵树、6 个 CPU 线程的历史测试中，CPU/Metal **训练时间中位数之比为 1.55 倍**。训练加预测的中位数之比为 1.53 倍；加入双方共用的一次性数据生成和 Dataset 构建时间后，估算整体比值为 1.45 倍。**这组数据来自大叶节点梯度预处理加入前的提交 <code>7807af1</code>。** 当前版本默认在大叶节点先由 GPU 聚合并量化选中行的梯度和 Hessian；其新旧路径对照见[公开基准报告](benchmarks/metal/README.md)。这些测试只来自一台机器，不能代表其他芯片或真实业务数据。

当前版本另有一组 330 万训练行、100 棵树的生成数据对照：旧／新 Metal 路径的训练时间中位数之比约 **1.27 倍**，模型及预测哈希相同。这是 Metal 内部算法对照，不是当前版本的 CPU/Metal 加速比。
在同规模的 100 棵树 CPU/Metal 配对测试中，当前版本的 CPU/Metal 训练时间中位数之比约 **1.67 倍**；轮数与上面的历史 500 棵树测试不同，详见报告。

当前版本还完成了 330 万训练行、20 万生成数据留出行、500 棵树的 CPU/Metal 交错对照：本次训练时间中位数之比为 **1.89 倍**。两种后端的第二次运行都明显变慢，因此这个数字不是稳定速度保证；与历史 500 棵树测试也不能直接相减。原始 JSON、预测差和限制见[公开基准报告](benchmarks/metal/README.md)。

运行同量级合成测试需较多统一内存，并且必须显式指定 <code>--full-scale</code>：

~~~bash
python examples/python-guide/metal_synthetic_benchmark.py \
  --full-scale --output synthetic_metal_result.json
~~~

要比较训练参数，可指定 <code>--max-bin</code>、<code>--num-leaves</code> 或 <code>--feature-fraction</code>；启用行采样时需同时设置 <code>--bagging-fraction</code> 和 <code>--bagging-freq</code>。脚本会记录配置、Dataset 构建时间、CPU/Metal 耗时与生成数据指标。不同配置的模型质量需另行验证。

## Star History

<a href="https://www.star-history.com/?repos=cuiyx35%2FLightGBM-Metal&type=date&legend=top-left">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/chart?repos=cuiyx35/LightGBM-Metal&type=date&theme=dark&legend=top-left" />
    <img alt="LightGBM-Metal 的 Star History" src="https://api.star-history.com/chart?repos=cuiyx35/LightGBM-Metal&type=date&legend=top-left" />
  </picture>
</a>

## 文档与二次开发

| 内容 | 中文 | English |
| --- | --- | --- |
| 项目入口 | 本页 | [README.en.md](README.en.md) |
| Metal 使用与验证 | [实验使用说明](docs/Metal-Experimental.md) | [Experimental Metal notes](docs/Metal-Experimental.en.md) |
| 架构与扩展流程 | [开发指南](docs/Metal-Development.md) | [Development guide](docs/Metal-Development.en.md) |
| 测试结果与限制 | [公开基准报告](benchmarks/metal/README.md) | [Benchmark report](benchmarks/metal/README.en.md) |
| AI 编码约束 | [AGENTS.md](AGENTS.md) | [AGENTS.en.md](AGENTS.en.md) |

## 来源与许可

本仓库从 LightGBM 4.7.0 的[官方提交 <code>8f7036f</code>](https://github.com/lightgbm-org/LightGBM/commit/8f7036f03627054d5a54a6f965b13f4b9ff2cb63)导入源码快照；更早的历史保留在[官方 LightGBM 仓库](https://github.com/lightgbm-org/LightGBM)。项目保留上游 [MIT 许可证](LICENSE)和第三方许可声明。上游文档、安装说明与通用 LightGBM 功能请以[官方文档](https://lightgbm.readthedocs.io/)为准。
