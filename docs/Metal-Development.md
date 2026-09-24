# Metal 实验后端开发指南

**简体中文** · [English](Metal-Development.en.md) · [返回项目首页](../README.md)

本指南面向开发此分支的人与 AI 编码工具。源码基于 LightGBM 4.7.0 的上游提交 <code>8f7036f</code>。本分支通过 CMake 选项 <code>USE_METAL</code> 增加 <code>device_type="metal"</code>。它仍是研究性后端，目前只在 Apple M5 上完成性能测量。

## 训练任务如何分工

~~~mermaid
flowchart LR
    A[LightGBM 分箱后的 Dataset] --> B[镜像符合条件的特征组]
    B --> C[Apple GPU 字节矩阵]
    D[梯度、Hessian、叶节点行索引] --> E[Metal 直方图计算]
    C --> E
    A --> F[CPU 计算其余组的直方图]
    E --> G[CPU 合并与 bin 校正]
    F --> H[CPU 分裂搜索与建树]
    G --> H
~~~

<code>MetalTreeLearner</code> 扩展了串行树学习器。<code>BuildMirror()</code> 选择只含一个特征、不是 multi-value、且 bin 数不超过 256 的组，并在共享内存的稠密字节矩阵中保存每个训练行的对应 bin。其他组仍由 CPU 处理。<code>ConstructHistograms()</code> 启动这些组的 Metal 计算，同时在 CPU 上构造其余直方图，然后等待 GPU 完成并把结果写入 LightGBM 的直方图数组。CPU 负责分裂搜索、树生长、模型序列化和预测。

GPU kernel 从源码中的 <code>kMetalSource</code> 字符串在运行时编译。每个 threadgroup 处理一个特征组和一片选定行。它把量化后的梯度与 Hessian 按有限长度的 chunk 累加到 32 位 threadgroup 原子变量，存储 64 位分片总和，最后由 CPU 转换并合并为双精度值。默认每 chunk 256 行、每 shard 32,768 行、每组 64 个线程。量化与浮点合并顺序可能改变临界分裂决策，因此不能保证 CPU 与 Metal 模型逐字节相同。

| 文件 | 职责 |
| --- | --- |
| <code>CMakeLists.txt</code> | 可选 Metal 构建及 Apple 框架链接 |
| <code>include/LightGBM/config.h</code>、<code>src/io/config.cpp</code> | 接受 <code>device_type=metal</code> |
| <code>src/treelearner/tree_learner.cpp</code> | 选择 <code>MetalTreeLearner</code> |
| <code>src/treelearner/metal_tree_learner.h</code> | 学习器接口和回退标志 |
| <code>src/treelearner/metal_tree_learner.mm</code> | 适用性判断、数据镜像、GPU kernel、CPU/GPU 并发、合并及 profiling |
| <code>examples/python-guide/metal_validate.py</code> | 五项合成数据模型级检查 |
| <code>examples/python-guide/metal_synthetic_benchmark.py</code> | 可复现的 CPU/Metal 耗时与合成任务质量报告 |

## 扩展特征组的步骤

1. 先确认目标特征组在 LightGBM 中的表示和 bin 布局。整个组只能由 CPU 或 GPU 中的一方负责，避免 CPU 结果覆盖 GPU 的 bin。
2. 同步修改镜像和 kernel。检查组内 bin 偏移、默认 bin、最高频 bin、缺失值及每组输出边界。
3. 保持 32 位 chunk 累加的上界有效。在把新组类型交给分裂搜索前，检查直方图守恒及最终 bin 校正。
4. 运行合成数据验证，对照串行诊断路径与并发路径。检查模型树和留出预测，不要只看短时间训练的耗时。
5. 在空闲机器上以目标行数、特征数和树数测试。记录芯片、线程数、预热、运行顺序、数据构建时间、训练与预测时间、峰值内存及相关模型指标。

微小预测差会在数百棵树中累积。合成测试中 CPU/Metal 预测几乎一致，只能说明工程检查通过，不能代表业务质量获批。即便总体指标上升，也应报告有意义的模型级差异。

## 本地构建与脚本入口

前置条件见[实验使用说明](Metal-Experimental.md)。在已配置 CMake 和 OpenMP 的 Mac 上：

~~~bash
bash tools/build-metal-macos.sh --validate
PYTHONPATH="$PWD/python-package" python3 examples/python-guide/metal_synthetic_benchmark.py \
  --output metal_quick_benchmark.json
~~~

确认 Python 加载的是当前仓库的 <code>lightgbm</code> 和 <code>lib_lightgbm.dylib</code>，避免其他已安装包遮蔽本后端。基准脚本默认快速测试；410 万行的完整规模需指定 <code>--full-scale</code>，且需要较多统一内存。两个脚本都支持 <code>--help</code>、失败时返回非零退出码，并写出 JSON；首先检查 <code>status</code>。基准报告包含 <code>configuration</code>、<code>run_order</code>、<code>runs</code>、<code>median</code>、速度比、预测差和限制；验证报告包含五个 <code>cases</code>。这些格式仍属实验性；修改字段时同步更新示例和文档。

## 诊断开关

| 环境变量 | 用途 |
| --- | --- |
| <code>LGBM_METAL_PROFILE=1</code> | 记录 GPU inflight/等待、CPU 直方图、准备、合并与 dispatch 耗时 |
| <code>LGBM_METAL_DISABLE_OVERLAP=1</code> | 串行执行 Metal 与 CPU 直方图计算，用于对照 |
| <code>LGBM_METAL_FORCE_CPU=1</code> | 在 Metal 学习器中强制使用 CPU 直方图，便于诊断 |
| <code>LGBM_METAL_COMPARE_HIST=1</code> | 将一个 GPU 直方图与 CPU 计算结果对照 |
| <code>LGBM_METAL_VERIFY_QUANTIZED_GROUP=all</code> | 对选定 dispatch 的 GPU 整数 bin 在 CPU 上重算 |
| <code>LGBM_METAL_VERIFY_QUANTIZED_DISPATCH=N</code> | 指定上一项要验证的 dispatch |
| <code>LGBM_METAL_THREADS_PER_GROUP=64&#124;128&#124;256</code> | 测试 kernel threadgroup 大小 |
| <code>LGBM_METAL_ROWS_PER_SHARD=...</code> | 测试每 shard 4,096 至 65,536 行；具体可选值见源码 |
| <code>LGBM_METAL_ROWS_PER_CHUNK=32&#124;64&#124;128&#124;256</code> | 测试量化累加 chunk 大小 |
| <code>LGBM_METAL_COMPACT_GROUPS=0</code> | 恢复旧的非紧凑 GPU grid，供 A/B 对照 |
| <code>LGBM_METAL_MIN_LEAF_ROWS=N</code> | 探索叶节点行数低于阈值时转 CPU |

这些开关可能改变结果和耗时。公开基准会拒绝诊断覆盖项，使主报告速度比使用正常路径。profiling 应单独运行，并明确标注为诊断任务。

## 公开发布与可移植性

这是独立实验分支，不是 LightGBM 官方版本。[公开基准](../benchmarks/metal/README.md)只使用一台 Apple M5 和生成数据。其他 M 系列芯片、OpenMP 配置、特征分布或树深均可能改变 CPU/GPU 的工作平衡。不要把真实应用数据及标识符放入本仓库。保留上游 MIT 许可和子模块声明。
