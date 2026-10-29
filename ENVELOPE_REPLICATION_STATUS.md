# A/C 复现实验执行记录

## 预注册

六个新父种子：101、202、303、404、505、606。Gaussian 父解从随机初始化经过 20000＋30000 步，再分叉 A/C，各追加 10000 步。旧三种子不进入主要验证。完整固定配方见 [说明](ENVELOPE_REPLICATION_GUIDE.md)。

## 本地验证

- 新增针对性测试 13 项通过，覆盖父阶段切换、两阶段断点恢复、A/C 分叉状态、独立动能、种子碰撞、互斥归因、方差分解、bootstrap、轴篡改和提交幂等。
- 实验固定源码 `1659b8ef33ade46dabe30a457f27bf915714c9da` 的严格 CUDA 套件为 309 通过，补充四项测试后为 **313 通过、0 失败、0 错误、0 跳过**，194.96 秒，见 [最终严格 CUDA 摘要](physics_evidence/replication_cuda_final_20261026/summary.json)。
- 当前分支提交 `407171a22406016f047040ed132af1f147a7af43` 的严格 CUDA 复验为 **314 通过、0 失败、0 错误、0 跳过**，224.91 秒，见 [当前摘要](physics_evidence/cuda_20261029T095400Z/summary.json)。
- 既有 12 份轨迹归因已完成；全部独立动能、log 振幅与符号审计通过。没有新增采样或训练更新，见 [归因报告](physics_evidence/residual_attribution_20261026/REPORT.md)。C/D 未触发四标志区域贡献约 73%–81% 的方差；各组均未观测到 >1000 的高抵消样本。以上是条件于已访问轨迹的描述统计。

## 正式训练与结果

正式 A/C 训练已按预注册完成，实验固定源码为 `1659b8ef33ade46dabe30a457f27bf915714c9da`。六个新父种子的 12 个 A/C case 全部完成并通过评估诊断；soft-exponential 组化学精度通过 **6/6**，Gaussian 组通过 **2/6**，配对平均改善 **2.198 mHa**，近似 95% 区间 `[-3.067, -1.330] mHa`。结果与证据边界见 [最终报告](physics_evidence/envelope_replication_final_20261028/REPORT.md)和[可复核摘要](physics_evidence/envelope_replication_final_20261028/verified_summary.json)。

## 本地工具记录

一次 `uv run ruff` 检查误触了环境同步，并重建了忽略目录 `.venv`；下载进程已终止，没有修改 uv.lock 或依赖配置。后续静态检查使用独立的 uvx ruff；全部训练验证与轨迹归因使用 F:\conda\envs\torch\python.exe，Conda 环境不受影响。如需再次使用项目 uv 环境，应重新执行所需的 uv sync。
