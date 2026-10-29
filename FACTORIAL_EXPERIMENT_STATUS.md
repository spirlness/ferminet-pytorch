# 裁剪 × 包络实验执行记录

2026-10-25（香港时间）。科学配置为 FermiNet 三种子、四组、每组 10000 次更新、每组 1048576 个正式评估样本。

**当前状态：正式 12/12 组完成，产物与统计复核通过。** 固定源码为 `9b001c5bbd4c143058248bbc0c6e3585412ae27c`，最后一组于香港时间 22:18:20 完成，22:24:13 确认 Kaggle COMPLETE。全部 350 个发布文件已取得，清单内 349 项 SHA-256、47 个源码文件以及 18 份 FP64 评估轨迹均核验通过。[最终报告](physics_evidence/factorial_final_20261025/REPORT.md)与[可复核摘要](physics_evidence/factorial_final_20261025/verified_summary.json)已归档。化学精度 A/B/C/D 分别为 2/3、3/3、3/3、3/3；三个预设配对效应的 Bonferroni 区间均跨零，改善证据不足。

## 第一轮实现与真实双 T4 短测

固定源码：`ae0bc0dc542b932004b7aa75a0766c10af3610c3`。

- 本地完整严格 CUDA 验收：296 项通过、0 失败、0 错误、0 跳过，197.32 秒；含 CPU、实际 CUDA 和编译检查。Bundled TCC 可用。[原始结果](physics_evidence/factorial_cuda_release_20261025/summary.json)、[环境与源码核对](physics_evidence/factorial_cuda_release_20261025/environment.json)。本地数量包括当时尚未提交、现已归档的冻结诊断测试，不等于远端运行时测试数量。
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


## 正式实验完成与产物核验

12 组各追加 10000 次更新，最终 checkpoint 均为相对 epoch 10000、累计 60000。12 份正式评估各保留 2048×512 个 FP64 样本，6 份零步评估各保留 512×512 个样本；无失败、缺组、超时或非有限能量。大型检查点、resume、起点状态与全部观察分片保留在 `campaign_runs/factorial_final_9b001c5/`。

使用 git archive 提取固定源码并运行其 `scripts.factorial_report`，349 个归档 SHA、父身份、更新数、独立评估种子、能量均值与 SEM 复算通过。补充核验逐片检查坐标、动能、势能、log 振幅、符号的 FP64 类型和有限性，时间与 walker 身份、E=T+V、轨迹逐值一致、ESS/R-hat/空间尾部统计均通过；Adam 与调度器净更新数、冻结适应及同父四组起点 walkers/RNG 也通过。

本次 A/B/C/D 化学精度通过数为 2/3、3/3、3/3、3/3，全部最终采样诊断和前后/极端点数值审计通过。裁剪、包络、交互作用的均值分别为 −0.116031、−1.046245、+0.164057 mHa，Bonferroni 近似区间分别为 [−0.394219, 0.162157]、[−2.606224, 0.513734]、[−0.617823, 0.945938] mHa；均跨零，按预设规则报告证据不足。零步替换包络使能量即时升高，不能用零步评估代替最终迁移表现。

[完整报告与边界](physics_evidence/factorial_final_20261025/REPORT.md)、[统计摘要](physics_evidence/factorial_final_20261025/verified_summary.json)、[原始轨迹审计](physics_evidence/factorial_final_20261025/raw_observation_audit.json)、[原始产物 SHA](physics_evidence/factorial_final_20261025/artifacts/archive_manifest.json)。此前运行日志快照保留，不代表最终状态。
