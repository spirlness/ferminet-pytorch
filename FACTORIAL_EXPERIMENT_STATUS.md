# 裁剪 × 包络实验执行记录

2026-10-25（香港时间）。科学配置为 FermiNet 三种子、四组、每组 10000 次更新、每组 1048576 个正式评估样本。

## 第一轮实现与真实双 T4 短测

固定源码：`ae0bc0dc542b932004b7aa75a0766c10af3610c3`。

- 本地完整严格 CUDA 验收：296 项通过、0 失败、0 错误、0 跳过，197.32 秒；含 CPU、实际 CUDA 和编译检查。Bundled TCC 可用。[原始结果](physics_evidence/factorial_cuda_release_20261025/summary.json)、[环境与源码核对](physics_evidence/factorial_cuda_release_20261025/environment.json)。本地数量包括已有未提交的冻结诊断测试，不等于远端运行时测试数量。
- 真实双 Tesla T4：[短测 version 1](https://www.kaggle.com/code/hwbwhzbsh/ferminet-factorial-smoke)。每卡 13 项运行时测试通过、无跳过。四组分别净新增 20 次更新；父 SHA、初始 walkers/RNG、前后数值审计、完整观察记录及日志核验均通过。
- 全部 55 个发布产物的 SHA-256 已核对。原始小文件见 [短测归档](physics_evidence/factorial_t4_smoke_20261025/verification.json)，大型检查点与观察分片位于该文件记录的本地下载目录。

| 测量 / 门槛 | 结果 |
|---|---:|
| 最慢组平均更新耗时 | 0.242730 秒 |
| 最慢组每条评估记录耗时，含观察分析 | 0.698289 秒 |
| 按实测外推的基础执行时长 | 24336.97 秒，约 6 小时 46 分钟 |
| 加 1.25 倍安全系数和 900 秒启动预留 | **31321.21 秒，约 8 小时 42 分钟** |
| 提交阈值 | 28800 秒，8 小时 |
| 第一轮门槛状态 | **正确性通过；预算未通过** |

[通过/拒绝回执](physics_evidence/factorial_t4_smoke_20261025/gate.json)与[流程记录](physics_evidence/factorial_t4_smoke_20261025/pipeline.json)确认第一轮没有提交正式任务。短测每组仅 64×512 个评估样本，不能据此报告新精度或选择实验组。

## 后续执行优化

在相同科学配置与预算门槛下，继续减少评估的小块 GPU 回传、复用动能坐标前向的 log 振幅/符号，并合并训练标量统计回传。正式样本数、训练步数、裁剪规则、包络、Adam、学习率调度和精确反向动能均保持原方案。优化版本必须重新经过完整 CUDA 验收及真实双 T4 短测；第一轮拒绝回执保留。

优化版本地完整严格 CUDA 验收为 **298 通过、0 失败、0 错误、0 跳过**，197.16 秒，见 [原始摘要](physics_evidence/factorial_cuda_io_20261025/summary.json)。本地冻结对照与 ABBA 测速的原始证据见 [对照说明](physics_evidence/factorial_io_comparison_20261025/README.md)。
