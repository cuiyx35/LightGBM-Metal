# Apple 芯片上的 Metal 训练实验

**简体中文** · [English](Metal-Experimental.en.md) · [返回项目首页](../README.md)

本仓库是基于 LightGBM 4.7.0 的独立研究分支，为 Apple Silicon macOS 添加 <code>device_type="metal"</code>。它不是 LightGBM 官方版本，目前只在 Apple M5 上测试过性能，尚未跨不同 M 系列芯片验证。独立仓库的历史从[上游提交 <code>8f7036f</code>](https://github.com/lightgbm-org/LightGBM/commit/8f7036f03627054d5a54a6f965b13f4b9ff2cb63)的源码快照开始；更早的历史仍在官方仓库。

架构和贡献流程见[Metal 开发指南](Metal-Development.md)。AI 编码工具还应阅读仓库的 [AGENTS.md](../AGENTS.md)。

Metal 路径为符合条件的特征组计算直方图。CPU 同时计算其他特征组，再执行分裂搜索及其余建树步骤。当前符合条件的组必须只含一个特征、不是 multi-value 组、bin 数不超过 256。预测使用标准 CPU 模型路径。Metal 直方图采用定点量化，树结构与预测可能不同于纯 CPU 训练；部署前应单独验证应用场景中的模型质量。

## 构建与使用

需要 Apple Silicon macOS、支持 Metal 的 GPU、CMake、C++ 工具链，以及用于 CPU 并行训练的 OpenMP 运行时。Python 基准和验证脚本还需要 NumPy、SciPy、pandas、scikit-learn、narwhals。使用 Homebrew 的典型步骤：

~~~bash
brew install cmake libomp
bash tools/build-metal-macos.sh --validate
~~~

脚本接受 <code>OPENMP_PREFIX</code>、<code>PYTHON_BIN</code> 环境变量和 <code>--jobs N</code> 参数。若 OpenMP 安装在非标准库路径，运行 Python 时还需把它的 <code>lib/</code> 目录加入 <code>DYLD_LIBRARY_PATH</code>。确认 Python 加载的是此检出目录中的包和启用 Metal 的 <code>lib_lightgbm.dylib</code>。训练参数示例：

~~~python
import lightgbm as lgb

params = {"objective": "binary", "device_type": "metal", "num_threads": 6}
model = lgb.train(params, training_dataset, num_boost_round=500)
model.save_model("model.txt")
~~~

返回 CPU 路径时设置 <code>device_type="cpu"</code>。预测和模型序列化仍使用 LightGBM 常规接口。

## 可脚本化的模型验证

[<code>examples/python-guide/metal_validate.py</code>](../examples/python-guide/metal_validate.py)在生成数据上运行五类模型级检查：稠密数值特征；带缺失值、类别、稀疏特征和 bagging 的混合输入；多直方图分片；常数 Hessian 情形下的 CPU 回退；CPU/GPU 并发与串行执行对照。脚本检查 CPU/Metal 预测、并发结果和模型重载，写出 JSON 报告；检查失败时退出码非零，并写入 <code>"status": "FAIL"</code>。这是冒烟测试，不要求应用模型的预测与 CPU 完全一致。

~~~bash
python examples/python-guide/metal_validate.py --output metal_validation.json
~~~

## 公开合成基准

[<code>examples/python-guide/metal_synthetic_benchmark.py</code>](../examples/python-guide/metal_synthetic_benchmark.py)按固定种子在本地生成所有特征和标签。默认运行 3 万训练行的快速冒烟测试。显式指定 <code>--full-scale</code> 时，规模为 330 万训练行、80 万留出行、512 个特征（150 个稠密数值特征和 362 个稀有二元特征）、500 轮 boosting。完整规模在一棵树的预热后，交替运行两次 Metal 与两次 CPU，复用同一个已构建的 Dataset。报告包含耗时、合成标签上的 AP/AUC、预测差、重复运行哈希和进程峰值 RSS。脚本不读取外部数据文件，报告不包含逐行样本或预测值。

~~~bash
python examples/python-guide/metal_synthetic_benchmark.py \
  --full-scale --output synthetic_metal_result.json
~~~

小规模冒烟测试省略 <code>--full-scale</code> 即可。完整规模需要较多统一内存；仅源码矩阵就约占 2 GiB。即使行数和特征数相同，生成数据的特征分布、特征捆绑、缺失模式和标签关系也会与真实应用不同。质量指标只对应这个生成任务。速度还取决于芯片型号、温度、后台负载和 OpenMP 配置。

一台 Apple M5 的完整规模结果及五项验证结果见[公开基准报告](../benchmarks/metal/README.md)。

本分支保留上游 [MIT 许可证](../LICENSE)及版权声明，第三方子模块保留各自许可文件。发布基准结果时，应同时说明确切提交、硬件、配置、运行顺序，以及是否计入数据构建时间。
