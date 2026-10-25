# 裁剪 × 包络实验执行记录

2026-10-25（香港时间）。科学配置为 FermiNet 三种子、四组、每组 10000 次更新、每组 1048576 个正式评估样本。

**当前状态：第三轮门槛通过，正式 12 组实验已提交。** 固定源码为 `9b001c5bbd4c143058248bbc0c6e3585412ae27c`，Kaggle 状态快照为 RUNNING；最新 17:07 日志快照已有 2/12 组完成训练和冻结评估，完整结果仍待收集核验。当前聊天已安排完成后的产物核验与报告收尾，运行期间状态不变时不发送例行通知。

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

## 第二轮真实双 T4 短测

固定源码：`d357eed4b0761442c1284ba98116fc95bb7304a9`。每组 100 次更新，每卡 14 项运行时检查通过、无跳过。四组正确性与全部 55 个发布文件哈希核验通过。最慢组更新均值 0.233545 秒，含观察分析的评估记录 0.658680 秒；使用相同 1.25 倍安全系数、900 秒预留和 8 小时阈值，预计 **29943.20 秒，约 8 小时 19 分钟**，预算仍未通过。见 [第二轮回执](physics_evidence/factorial_t4_smoke_io_20261025/gate.json)及[归档核验](physics_evidence/factorial_t4_smoke_io_20261025/verification.json)。正式 12 组配置哈希与第一轮一致，没有提交正式任务。

继续优化已经验证为全有限的训练批次：避免重复检查与动态布尔索引，保留部分非有限数据的原有拒绝/警告/裁剪行为。仍须通过完整严格 CUDA 验收和新的真实双 T4 门槛。

全有限路径优化后的本地严格 CUDA 验收为 **300 通过、0 失败、0 错误、0 跳过**，185.47 秒，见 [原始摘要](physics_evidence/factorial_cuda_fast_loss_20261025/summary.json)。四组临时更新与基线的逐值回归见 [训练路径对照](physics_evidence/factorial_loss_fast_20261025/parity.json)。

## 第三轮真实双 T4 门槛通过与正式提交

固定源码：`9b001c5bbd4c143058248bbc0c6e3585412ae27c`。每卡 **15 项运行时检查通过、无跳过**；四组各净新增 **100 次更新**。父状态/RNG、前后数值审计、完整观察分片、日志和全部 55 个发布文件哈希均核验通过。

最慢组更新均值 **0.213749 秒**，含观察分析的评估记录 **0.625056 秒**。沿用相同的 1.25 倍安全系数与 900 秒启动预留，正式 12 组预计 **27863.00 秒，约 7 小时 44 分钟**，低于 28800 秒门槛。这是短测外推，不保证实际时长。没有缩短正式训练或评估，没有修改网络、优化器或动能算法；正式配置哈希在三轮中始终为 `96ad8912f83ea9d064eac193199970b48b5cce927508d60c6a12d5665448cd78`。

- [门槛通过回执](physics_evidence/factorial_t4_smoke_passed_20261025/gate.json)
- [原始产物核验与下载位置](physics_evidence/factorial_t4_smoke_passed_20261025/verification.json)
- [自动提交流程记录](physics_evidence/factorial_t4_smoke_passed_20261025/pipeline.json)
- [正式任务状态快照](physics_evidence/factorial_t4_smoke_passed_20261025/long_status_snapshot.json)
- [正式 Kaggle 任务](https://www.kaggle.com/code/hwbwhzbsh/ferminet-factorial-accuracy)，version 1。

Kaggle 返回 RUNNING。随后通过 Kaggle CLI 支持的实时日志流取得了训练日志，最新快照见下节。正式结果须待 12 组训练和独立评估结束后验证，最终报告将写入 `physics_evidence/factorial_final_20261025/REPORT.md`；目前没有新的精度结论。

## 实时训练日志快照（香港时间 16:03）

已通过 `KaggleApi.kernels_logs_stream` 拉取 531 条服务端日志事件。两张 T4 均在运行，worker exit_code 均为 null；未见 Traceback、case_failed、NaN/Inf 或显存不足。

| GPU | 实验 | 新增更新 / 10000 | 累计轮数 | 最新训练批次能量 Ha |
|---|---|---:|---:|---:|
| 0 | ferminet_seed22_C | 4547 | 54547 | -1.17644596 |
| 1 | ferminet_seed11_D | 4505 | 54505 | -1.17410445 |

当前完成 0/12 组；其他组等待各自 GPU 的固定执行顺序。以上能量来自训练批次，不能用于化学精度判定，须等待正式独立 FP64 冻结评估。

[完整日志](physics_evidence/factorial_live_20261025T080331Z/console.log)；[结构化进度](physics_evidence/factorial_live_20261025T080331Z/progress_summary.json)；[采集来源和哈希](physics_evidence/factorial_live_20261025T080331Z/capture_meta.json)。

## 实时训练日志快照（香港时间 17:07）

当前已完成 **2/12** 组：C / seed 22 于 16:49:07 完成，D / seed 11 于 16:49:57 完成。两组均已结束 10000 次追加更新及冻结评估流程；日志未提供完整能量/SEM/诊断结果，因此此处不报告化学精度判定。

| GPU | 当前实验 | 新增更新 / 10000 | 最新训练批次能量 Ha |
|---|---|---:|---:|
| 0 | ferminet_seed33_A | 3343 | -1.17199767 |
| 1 | ferminet_seed33_C | 2809 | -1.17363381 |

Kaggle 仍为 RUNNING；最新心跳显示两 worker 正常存活、退出码均为 null，GPU 利用率约 32%–33%、显存约 177–179 MiB。本次 1625 条日志事件未见 Traceback、NaN/Inf、OOM 或 case_failed。训练批次能量不能替代独立评估精度。

[完整日志](physics_evidence/factorial_live_20261025T090743Z/console.log)；[结构化进度](physics_evidence/factorial_live_20261025T090743Z/progress_summary.json)。
