# AI 编码工具指南

**简体中文** · [English](AGENTS.en.md)

本仓库是独立的 LightGBM 实验分支。Metal 后端仅在 Apple Silicon macOS 上运行。修改后端前，先阅读[Metal 开发指南](docs/Metal-Development.md)。当 <code>USE_METAL=OFF</code> 时，保持上游 LightGBM 行为不变。

## 本分支的工作范围

- 主要入口为 <code>CMakeLists.txt</code>、<code>include/LightGBM/config.h</code>、<code>src/io/config.cpp</code>、<code>src/treelearner/tree_learner.cpp</code> 和 <code>src/treelearner/metal_tree_learner.{h,mm}</code>。
- Metal 学习器按**整个特征组**决定交给 GPU 还是 CPU。CPU 与 GPU 并发构造直方图，分裂搜索仍在 CPU 上。扩展支持的组类型时，保留这一所有权规则。
- 源码、示例和基准报告不得包含私有数据集、私有特征名、逐行预测、凭据或本机路径。保留上游和第三方许可证声明。

## 检查改动

1. 在单独的 CMake 构建目录中使用 <code>-DUSE_METAL=ON</code> 编译。在 macOS 上配置可用的 OpenMP 运行时，以支持 CPU 并行训练。
2. 运行 <code>examples/python-guide/metal_validate.py --output /tmp/metal_validation.json</code>。五项合成数据测试覆盖混合特征处理、回退、模型重载及 CPU/GPU 并发；确认 <code>status</code> 为 <code>PASS</code>。
3. 运行 <code>examples/python-guide/metal_synthetic_benchmark.py --output /tmp/metal_benchmark.json</code> 做快速 CPU/Metal 冒烟测试。仅在预先安排且机器空闲时用 <code>--full-scale</code> 做完整性能比较。
4. 对数值或性能改动，使用相同输入与参数，记录 CPU/Metal 运行顺序和预热，检查模型与预测哈希，并同时记录速度和质量差异。

Python 脚本写出汇总 JSON，失败时以非零状态退出。它们需要本分支的 Python 包和启用 Metal 的原生库；解读基准前先确认实际导入路径。profiling 环境变量会带来额外开销，主报告计时应关闭。

Star History 图使用 <code>api.star-history.com</code> 的公开嵌入链接；不要在 README、链接或仓库配置中加入个人访问令牌。

不要推送到指向 LightGBM 官方仓库的 <code>upstream</code> 远端。公开项目只推送到仓库所有者账号下的 <code>origin</code>，推送前复核目标和提交署名。公开源码基准只能使用生成数据。
