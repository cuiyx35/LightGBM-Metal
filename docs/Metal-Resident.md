# GPU 常驻建树实验

**简体中文** · [English](Metal-Resident.en.md) · [返回首页](../README.md)

## 启用

在启用 <code>USE_METAL</code> 的本分支中，设置环境变量后正常调用 LightGBM：

~~~python
import os
import lightgbm as lgb

os.environ["LGBM_METAL_RESIDENT"] = "1"
model = lgb.train(
    {"objective": "binary", "device_type": "metal", "num_threads": 6},
    train_set,
    num_boost_round=100,
)
~~~

不设置该变量或设为 <code>0</code>，使用默认 CPU/GPU 协同路径。环境变量在创建学习器时读取，不能用于控制同一进程中并发创建的不同模型。预测、保存和加载仍使用标准 LightGBM 接口。

## GPU 与 CPU 的工作

符合条件的一棵树在一个 Metal 命令缓冲区内完成直方图、分裂候选、最优叶选择和行分区。GPU 保存整棵树的叶队列与直方图，只计算较小孩子的直方图，较大孩子由父直方图相减得到。每棵树完成后，CPU 将分裂轨迹转换为常规 LightGBM 模型，并接收最终行分区。

目标函数、每轮梯度/Hessian、采样、训练分数更新和模型对象仍由现有 CPU 训练器处理。首次训练时，CPU 将 Dataset 的实际特征 bin 展开为 GPU 字节矩阵；bundled、稀疏和 multi-value 组也按实际特征展开。二值特征打包处理，稀疏非默认行有独立索引。

## 支持与回退

| 项目 | 当前范围 |
| --- | --- |
| 分箱与叶数 | 每特征最多 256 bin，最多 256 叶 |
| 分裂 | 数值、类别 one-hot 与类别集合、None/Zero/NaN 缺失路由 |
| 约束与正则 | 最小行数/Hessian/增益、最大深度、L1/L2、max_delta_step、path_smooth、类别正则参数 |
| 已测试训练流程 | 二分类、回归、多分类、需要叶输出更新的 regression_l1、bagging、GOSS、按树采样特征、叶池大小调整、保存后续训 |
| 明确回退 | 超过 bin/叶数上限、单调/交互约束、feature_contri、CEGB、extra_trees、按节点特征采样、forced splits、上游量化梯度、linear_tree |
| 内存 | 展开矩阵和工作缓冲区的保守估算超过 4 GiB 时回退；Dataset、Python 数据和临时内存另计 |

回退日志包含 <code>resident tree fallback:</code>，成功使用常驻路径包含 <code>resident tree:</code>。回退使用现有学习器，不丢弃配置约束。<code>LGBM_METAL_FORCE_CPU=1</code> 优先强制 CPU。

## 数值和质量检查

每棵树把梯度/Hessian 定点量化到最大绝对值 <code>2^20</code>。每个最多 1,024 行的 chunk 使用 32 位线程组原子累加，最坏绝对和不超过 <code>2^30</code>；分片、逐叶直方图和父子相减使用 64 位整数。分裂评分使用 GPU 单精度浮点，临近并列的候选可能与 CPU 不同。

运行生成数据验证：

~~~bash
PYTHONPATH="$PWD/python-package" python3 examples/python-guide/metal_resident_validate.py \
  --output metal_resident_validation.json
~~~

验证会在原生代码中按每个 GPU 分裂重放 CPU 行路由，核对最终行集合，并调用 CPU 的 FeatureHistogram 检查所选特征的最优增益。同时检查缓存训练分数、标准预测、模型重载、生成留出指标和回退行为。诊断开销不用于性能结论。

直接更换含混合稀疏特征的 Dataset 曾在 CPU 基线触发空孩子断言，该流程尚未验收。需要换数据时，优先新建 Booster，以保存的模型作为 <code>init_model</code> 续训。数值 Dataset 更换已在验证中覆盖。生成数据验证不能代替应用质量评估。

## 性能复测

~~~bash
PYTHONPATH="$PWD/python-package" python3 examples/python-guide/metal_resident_benchmark.py \
  --output metal_resident_quick.json
# 330 万训练行、20 万生成留出行、512 特征、100 棵树、6 线程
PYTHONPATH="$PWD/python-package" python3 examples/python-guide/metal_resident_benchmark.py \
  --full-scale --output metal_resident_full.json
~~~

脚本使用同一 Dataset，先各预热一棵树，再按常驻→协同→CPU→CPU→协同→常驻顺序训练。fit 包含原生初始化和镜像，不含共享的数据生成与 Dataset 构建；每次训练和预测耗时单独记录。JSON 包含源码/库哈希、模型/预测哈希、AP/AUC/log loss、预测差和负载。<code>PASS</code> 表示生成数据质量及重复性检查通过，不表示速度一定更快。

## 二次开发

实现入口为 <code>src/treelearner/metal_resident_tree.{h,mm}</code>；<code>MetalTreeLearner::Train()</code> 负责选择路径，<code>DataPartition::SetLeafPartition()</code> 接收最终行池。修改支持范围时同时检查回退、参数重置、直方图守恒和 CPU 模型接口。

| 诊断变量 | 用途 |
| --- | --- |
| <code>LGBM_METAL_RESIDENT_VERIFY=1</code> | 原生 CPU 分裂/行集合复核 |
| <code>LGBM_METAL_PROFILE=1</code> | 初始化、量化、编码、GPU 和模型材料化累计时间 |
| <code>LGBM_METAL_RESIDENT_PROFILE_STAGES=1</code> | 每个 kernel 单独提交并等待；仅用于定位，改变执行方式 |
| <code>LGBM_METAL_RESIDENT_INDIRECT=0</code> | 关闭设备端间接 dispatch |
| <code>LGBM_METAL_RESIDENT_SKIP_DEFAULT=0</code> | 关闭最频 bin 省略与守恒重建 |
| <code>LGBM_METAL_RESIDENT_SPARSE=0</code> | 关闭稀疏行索引扫描 |
| <code>LGBM_METAL_RESIDENT_BINARY=0</code> | 关闭二值特征打包 |
| <code>LGBM_METAL_RESIDENT_STABLE=0</code> | 使用原子分区代替稳定分区 |
| <code>LGBM_METAL_RESIDENT_STAGE=0</code> | 关闭选中叶梯度连续化 |
| <code>LGBM_METAL_RESIDENT_CHUNK=256&#124;512&#124;1024</code> | 改变原子累加 chunk；默认 1,024 |

所有布尔诊断变量只接受 <code>0</code> 或 <code>1</code>。性能脚本拒绝外部 Metal 覆盖变量，避免混淆路径。Objective-C++ 文件也必须带 OpenMP 编译参数，否则 CPU 端的特征展开与量化循环会串行执行。
