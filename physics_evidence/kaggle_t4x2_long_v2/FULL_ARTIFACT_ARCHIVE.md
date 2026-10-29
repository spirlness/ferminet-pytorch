# 完整训练产物本地归档

2026-10-22 已下载 `hwbwhzbsh/ferminet-t4x2-multiseed` version 2 的全部 **61 个远端产物，共 148,713,756 字节**。训练源码固定为 `d2618dfb6780f1da49f0bb4ab247d29ec15e41f8`。

## 本地位置

- 压缩归档：`campaign_runs/archives/kaggle_t4x2_long_v2_20261022.zip`，**89,819,081 字节**（约 89.8 MB）。
- 归档 SHA-256：`campaign_runs/archives/kaggle_t4x2_long_v2_20261022.zip.sha256`。
- 已解压目录：`campaign_runs/kaggle_artifacts_v2/`；使用说明位于该目录的 `README.md`。

压缩包 SHA-256：

```text
c83c62708b3d843e11b9a15cd1b9f0b641e4d31da0ad939bafa1427a796eb165
```

这些 `campaign_runs/` 链接指向本机已下载的产物，目录受 `.gitignore` 忽略，**不会随 Git 克隆或文档提交分发**。分享完整产物应单独传输 ZIP 与 SHA-256 文件；本目录中的小型回执和文件清单可随仓库保存。

## 归档内容

| 内容 | 数量或说明 |
|---|---|
| `final_checkpoint.pt` | 6 份最终模型状态；均为原计划第 20000 轮 |
| `resume.pt` | 6 份含 Adam、余弦调度、阶段、walkers、采样计数及 Torch CPU/CUDA RNG 的状态 |
| `evaluation_trajectory.pt` | 6 份原始 FP64 局域能量轨迹，每份 `(2048,512)`，共 6,291,456 个样本 |
| 训练历史与结果 | 六组完整 `training_history.json`、`result.json`、前后精度审计、progress、manifest |
| 日志与环境 | 两个 worker 日志、两卡运行时检查、配置、环境、最终摘要，以及额外拉取的完整控制台日志 |
| 固定源码 | `source_d2618df.zip`；关闭 Windows 自动换行转换，源码文件哈希全部与远端 campaign 清单匹配 |
| 复核资料 | 下载/核验脚本、固定版本统计代码、逐文件哈希、核验结果及最终报告快照 |

包内报告是归档前的文档快照；本次新增的原始轨迹复算证据以 `artifact_validation.json` 为准。

## 已完成的核验

1. 远端 61 个文件全部下载，HTTP Content-Length 与本地大小匹配，并计算本地 SHA-256。服务未返回逐文件 MD5，未声称完成所有产物的服务端哈希比对。
2. 六份最终 checkpoint 的 SHA-256 与远端各组 `result.json` 公布值匹配。12 份 checkpoint 均可通过 `torch.load(..., map_location="cpu", weights_only=True)` 读取，schema 为 2、epoch 为 20000、stage 为 2。
3. 同组最终与续训 checkpoint 的模型张量一致；训练历史与 `training_history.json` 完全匹配，优化器、调度器及 CPU/CUDA RNG 状态存在，walkers 形状及有限性检查通过。
4. 六份轨迹均为 FP64、形状 `(2048,512)`、所有观测有限。使用固定源码的 `energy_statistics` 在本地 CPU / PyTorch 2.14.0 复算，均值、方差、SEM、ESS、split-Rhat、相关时间、分块平台与化学精度判定匹配原始结果；数值比较采用相对容差 `1e-9`、绝对容差 `1e-12`。
5. ZIP 中各负载文件逐个通过大小和 SHA-256 核验，ZIP CRC 检查通过。

持久记录：[归档回执](full_artifact_receipt.json)、[产物核验详情](full_artifact_validation.json)、[远端产物清单及本地哈希](full_artifact_inventory.json)、[ZIP 负载清单](full_artifact_manifest.json)。

原环境为 Kaggle PyTorch 2.11.0+cu128 / CUDA 12.8；此次核验没有执行续训，也不保证跨硬件或版本逐位恢复。六组已到原训练上限，若要延长训练，需明确迁移配置与调度。使用同一统计算法复算不构成采样遍历性或物理实现的独立证明，精度结论保持不变：FermiNet 1/3 达标，Psiformer 0/3 达标。
