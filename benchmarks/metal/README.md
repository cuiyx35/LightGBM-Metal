# Apple M5 合成数据基准

**简体中文** · [English](README.en.md) · [返回项目首页](../../README.md)

这些报告来自一台配备 **32 GiB 统一内存**、接通电源的 Apple M5 Mac，使用本仓库的实验性 Metal 后端。完整规模基准与首份剖析对应的独立仓库源码状态为提交 <code>7807af1</code>；GPU 执行时间复核只在同一内核上增加了诊断计时。CPU 和 Metal 测试使用同一份预先构建的 Dataset，以及启用 Metal 的 LightGBM 4.7.0 检出目录。完整规模基准完成后，还用 CMake 4.4.3 和 Apple Clang 21 在全新构建目录中重新配置、编译，并针对重建后的库再次通过五项验证。

| 报告 | 工作负载 | 结果 |
| --- | --- | --- |
| [完整规模基准](m5_synthetic_4m_500trees.json) | 330 万训练行、80 万留出行、512 个特征、500 棵树、6 个 CPU 线程 | CPU/Metal 训练时间中位数之比 **1.55 倍** |
| [模型级验证](m5_validation.json) | 五项小规模合成数据测试 | 全部通过 |
| [瓶颈剖析](m5_profile_3m_100trees.json) | 330 万训练行、100 棵树、6 线程、开启 profiling | GPU 命令在途累计 16.14 秒；CPU 等待 GPU 累计 12.92 秒 |
| [GPU 执行时间复核](m5_gpu_execution_3m_100trees.json) | 相同配置的新一次诊断运行 | Metal 报告 GPU 实际执行累计 14.58 秒；命令在途累计 15.87 秒 |
| [大叶节点梯度预处理对照](m5_stage_selected_3m_100trees.json) | 330 万训练行、20 万生成数据留出行、512 个特征、100 棵树、6 线程；旧／新 Metal 路径交错运行 | 旧／新训练时间中位数之比 **1.27 倍**；模型与预测哈希相同 |
| [当前版本 CPU/Metal 对照](m5_current_3m_100trees_cpu_metal.json) | 330 万训练行、20 万生成数据留出行、512 个特征、100 棵树、6 线程；Metal/CPU 交错运行 | CPU/Metal 训练时间中位数之比 **1.67 倍** |
| [当前版本 500 棵树 CPU/Metal 对照](m5_current_3m_500trees_cpu_metal.json) | 同一生成器、330 万训练行、20 万留出行、500 棵树、6 线程；Metal/CPU 交错运行 | CPU/Metal 训练时间中位数之比 **1.89 倍**，重复运行有明显耗时漂移 |

完整规模测试按固定种子生成所有特征和标签，其中有 150 个稠密数值特征和 362 个稀有二元特征。CPU 和 Metal 分别进行一棵树的预热后，运行顺序是 **Metal → CPU → CPU → Metal**。训练耗时（秒）为 Metal **108.00 / 125.67**，CPU **178.93 / 183.50**。训练加预测的中位数为 Metal **121.35** 秒、CPU **185.85** 秒，比值为 **1.53 倍**。一次性数据生成和 Dataset 构建耗时 **21.72** 秒，不计入各后端单次耗时；若将同一份共用成本分别加入两个后端，估算整体比值为 **1.45 倍**。进程峰值 RSS 为 **9.72 GiB**。

每个后端重复运行得到的模型和预测哈希各自一致。在生成数据的留出预测上，Metal/CPU 最大差值为 **1.86e-6**，AP 差值为 **2.07e-9**。这些数字不能说明应用模型的质量或预测差异。生成数据没有再现真实的特征分布、缺失模式、类别值和标签关系。结果只有一个芯片、一个随机种子、每个后端两次运行；温度和后台负载可能改变耗时。模型级验证的耗时不是性能基准。

表中 **1.55 倍**的 500 棵树 CPU/Metal 比值来自优化前的 <code>7807af1</code>。当前分支默认只在选中行数超过一个分片时，先由 GPU 聚合并量化梯度与 Hessian。新增的 Metal 对照使用固定种子的生成数据，顺序是**旧→新→新→旧**，训练秒数分别为 **21.945 / 17.528 / 17.697 / 22.851**；旧／新中位耗时分别为 **22.398 / 17.612** 秒，旧／新比值约 **1.27**。这是同一 Metal 模型的两种计算路径对照，不是新的 CPU/Metal 加速比；没有包含数据生成、Dataset 构建和预测时间。两种路径在这份生成数据上的模型文本与留出预测哈希完全相同。机器温度与后台任务仍可能改变结果。

当前版本的 100 棵树 CPU/Metal 对照复用相同生成器与 Dataset，顺序为 **Metal→CPU→CPU→Metal**。训练秒数为 Metal **17.253/18.225**、CPU **28.941/30.296**；训练中位 **17.739/29.618** 秒，比值约 **1.67**。训练加预测中位比值约 **1.66**。生成数据留出预测的 Metal/CPU 最大差 **3.17e-6**，超过 0.001 的条数为零。它与上面的 500 棵树测试轮数及留出集大小不同，不应用两个比值推断跨版本提速；两者都不能说明真实应用模型的质量。

当前版本 <code>672b9d2</code> 的另一组 **500 棵树**对照仍用 330 万训练行、512 特征和 6 线程，留出集为 20 万生成行。Metal→CPU→CPU→Metal 的训练秒数依次为 **78.393 / 153.729 / 182.691 / 99.807**；Metal 与 CPU 的中位时间分别为 **89.100 / 168.210** 秒，比值 **1.89**，训练加预测比值 **1.88**。两种后端的第二次运行都明显变慢，因此这只是本次交错测试的描述值，不能视为稳定加速保证。每个后端内部的模型和预测哈希各自一致；Metal/CPU 留出预测最大差 **3.17e-6**，没有超过 0.001 的行。峰值 RSS **9.408 GiB**。它与历史 500 棵树测试的留出集大小、源码和运行时段不同，两个比值不能直接相减来表示改版增益。

使用当前分支的 Python 包和启用 Metal 的原生库构建后，可按以下命令复现：

~~~bash
python examples/python-guide/metal_validate.py \
  --output metal_validation.json
python examples/python-guide/metal_synthetic_benchmark.py \
  --full-scale --output metal_synthetic_benchmark.json
python examples/python-guide/metal_stage_benchmark.py \
  --full-scale --output metal_stage_benchmark.json
python examples/python-guide/metal_synthetic_benchmark.py \
  --train-rows 3300000 --held-rows 200000 --features 512 \
  --dense-features 150 --rounds 100 --threads 6 --repeats 2 \
  --output metal_current_cpu_metal.json
python examples/python-guide/metal_synthetic_benchmark.py \
  --train-rows 3300000 --held-rows 200000 --features 512 \
  --dense-features 150 --rounds 500 --threads 6 --repeats 2 \
  --output metal_current_500_cpu_metal.json
~~~

脚本只写出汇总 JSON 和哈希；本目录不包含外部数据文件或逐行预测。

### 瓶颈剖析说明

剖析报告使用同一生成器，但只训练 100 棵树，单次运行且开启了额外计时。首份报告中学习器内部训练累计 19.58 秒，其中 GPU 命令在途 16.14 秒；CPU 直方图累计 3.22 秒、分裂搜索 0.49 秒、数据镜像 0.62 秒。增加 GPU 时间戳后的另一次运行，6,200 次 GPU 命令均取得有效时间戳：实际执行累计 14.58 秒，提交到完成的墙钟时间累计 15.87 秒。两次运行不能逐项相减比较。GPU 与 CPU 的部分工作并行，等待时间也包含在命令在途时间中，**这些时间不能相加**。这支持继续研究 GPU 直方图内核；它不是独立的提速结果，也不能代表其他数据。判断性能请以上面的未开启 profiling 的交错基准为准。
