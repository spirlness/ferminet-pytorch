# Kaggle 长期 version 2 原始证据

任务：`hwbwhzbsh/ferminet-t4x2-multiseed` version 2；固定训练源码为 `d2618dfb6780f1da49f0bb4ab247d29ec15e41f8`。任务在 2026-10-22 显示 `COMPLETE`，六组均完成训练与冻结评估。说明和结论见[最终实验报告](../../KAGGLE_T4_FINAL_RESULTS.md)。

本目录由已下载的 `campaign_runs/kaggle_complete_v2/` 归档而来：

- `ferminet_results/summary.json`、六份 `result.json` 为实际输出，未经改写，含各组采样诊断、精度审计和 checkpoint 哈希。
- `campaign.json`、`bootstrap.json` 及六份 `manifest.json` 保留环境、配置、源码身份与源码文件哈希。
- `runtime_tests_gpu0.log`、`runtime_tests_gpu1.log` 为两卡各 3 项运行时检查记录。
- `console.log` 为 Kaggle `kernels_logs` 返回的日志条目解码后的完整文本；不是重新生成的模拟训练日志。
- `archive_manifest.json` 记录上述文件的 SHA-256 和大小；它是本地生成的归档清单，不是 Kaggle 签名或独立认证。
- `documentation_validation.json` 是本次文档更新时的派生核验记录，验证归档哈希、六组完成状态、统计数值、文档表格及本地 Markdown 链接；没有重新运行训练或从轨迹复算统计。
- 后续已完成全部训练产物的本地下载和压缩归档，含 checkpoint、原始评估轨迹及固定源码；位置、范围和 SHA-256 见[完整产物归档说明](FULL_ARTIFACT_ARCHIVE.md)。新增的 `full_artifact_receipt.json`、`full_artifact_validation.json`、`full_artifact_inventory.json`、`full_artifact_manifest.json` 分别保留回执、原始轨迹复算记录、远端产物清单和 ZIP 负载清单；它们不属于最初 18 个原始证据文件的 `archive_manifest.json`。

训练历史、独立 worker 日志、checkpoint 和评估轨迹 `.pt` 未直接纳入本目录；它们已保存在受 Git 忽略的本地完整归档中，也可从原 Kaggle 私有任务下载。归档说明明确区分本机文件链接与可随仓库保存的持久证据。结果文件中的 `/kaggle/working/...` 仍是远端原始路径，本地对应文件在归档的 `ferminet_results/` 下。
