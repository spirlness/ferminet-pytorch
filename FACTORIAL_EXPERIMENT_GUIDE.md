# 裁剪 × 包络受控实验

本实验预先固定 H₂、R=1.4 bohr、单重态，使用 FermiNet 三个种子 11/22/33 的 50000 轮最终检查点。每个父检查点分为四组，每组净新增 10000 次更新，总计 12 次实验。比较的是已有 Gaussian 解的迁移，不能推广为从头训练、Psiformer 或其他分子的结论。

| 组别 | 包络 | 裁剪 |
|---|---|---|
| A | Gaussian | median-MAD + tanh，保留原规则 |
| B | Gaussian | 标准 median + mean absolute deviation + hard |
| C | soft_exponential，L=2 bohr | 与 A 相同 |
| D | soft_exponential，L=2 bohr | 与 B 相同 |

## 物理与状态

新包络使用 `exp(-alpha*f(r))`，`f(r)=2*r*r/(sqrt(1+(r/L)^2)+1)`。近核 `f(r)=r²+O(r⁴)`，远场径向斜率趋于 `2L`；它与显式核 cusp Jastrow 兼容。L 固定而非可学习，参数数量与 Gaussian 相同。本实验固定比较这些显式分支；2026-10-28 起，启用 cusp Jastrow 且未指定包络时的默认值已改为 soft_exponential，Gaussian 和 exponential 仍可作为显式配置使用。

`clip_method="mean_hard"` 的窗口中心为标准中位数（偶数样本取中间两数均值），半宽为 `clip_factor*mean(abs(E-center))`，硬裁剪后围绕裁剪均值中心化。没有依赖能量零点的下限；常量批次保持原值。这是整套规则的对照，不能区分尺度与硬裁剪的单独因果作用。规则参考[固定版本官方实现](https://github.com/google-deepmind/ferminet/blob/e9d5a604d15c617864e7caefd948d87453b01ae6/ferminet/loss.py)。

`VMCDriver.fork_from_checkpoint(path, expected_sha256=..., parent_model_identity=...)` 明确创建新实验：核验父 SHA、模型身份、核位置/电荷、spin partition、walker dtype/形状及 CPU/CUDA 后端，复制权重和 walkers，恢复父 RNG，统一清空 Adam、调度器与历史。实验轮数从零开始，`experiment_origin` 记录父轮数及配置差异。旧 schema-2 父文件没有模型身份时，必须提供与固定父源码相对应的显式身份。普通 `load_checkpoint` 和既有 `extend_from_checkpoint` 不放宽恢复约束。

四组均训练全部参数，不冻结 Jastrow。主干/Jastrow 峰值学习率 1e-4/1e-5，500 步预热、10000 步逆时间衰减、最低比例 0.1。512 walkers、10 步训练 MCMC、梯度范数上限 1、FP32 和精确反向动能。四组统一 eager，无 AMP、TF32 或随机动能。每个种子的四组从相同父 walkers/RNG 开始，各自预热 1000 步；生产阶段冻结提议适应。

## 正式评估与判定

每个种子、每种包络共享一份零更新评估：512 walkers × 512 条记录。它描述替换包络的即时影响，不代替训练后的精度证据。最终 12 组均使用预先固定且互不重复的独立种子进行 FP64 冻结评估：512 walkers × 2048 条记录、1000 步预热、每条记录推进 20 步、能量批次 128。

`evaluate_frozen_wavefunction(..., observations_path=..., records_per_shard=128)` 增加可选的分片观察记录。每片保留 `time_index`、`walker_id`，以及 `positions`、原始 `energy`、`kinetic`、`potential`、`log_abs`、`sign`。manifest 包含 SHA-256 和完整状态。异常/中断会留下 `complete=false` 的可核查分片；正式评估拒绝非有限观察，不删除尖峰。真实库仑距离独立计算，不使用平滑特征距离。

观察记录复用 `compute_local_energy(..., return_wavefunction=True)` 的坐标前向，额外返回 detached log 振幅与符号；默认仍返回 E/T/V 三项。这些 detached 值不能用于参数代理损失。每条时间记录在 GPU 上合并指标，再集中回传，减少小块同步；能量算法、分批大小和采样调用顺序不变。复用求导前向的 log 振幅与额外 no-grad 前向可能存在浮点舍入差异，必须通过数值审计，不宣称全部观察字段逐位相同。

主要报告参考偏差、SEM、混合诊断、化学精度、空间尾部方差/能量贡献和极端样本数值审计。化学精度要求 `abs(E-E_ref)+1.96*SEM <= 1.5936 mHa`，且采样诊断通过；参考能量为 -1.174475714220 Ha。重尾下区间是近似值。

对每个父种子计算以下配对效应，再在三个种子间比较：

- 裁剪：`(B-A+D-C)/2`。
- 包络：`(C-A+D-B)/2`。
- 交互：`D-C-B+A`，表示非加性，不能独立视为某个方案的总收益。

三个效应采用 Bonferroni 调整的近似区间，总 alpha=0.05、Student t 自由度 2；标准误取种子间估计与独立评估 SEM 传播值的较大者。平均降低至少 0.2 mHa 且区间上界小于零，才报告预设改善证据。缺组、失败、数值审计或采样诊断不足时，不发布完整交叉实验结论。三个种子不足以证明普遍稳定性。

## 执行与门槛

正式配置为 `configs/factorial_t4x2.json`，短测为 `configs/factorial_smoke.json`。短测保留正式模型规模、512 walkers、1000 步预热及四种组合；每组 100 次更新、64 条最终评估记录，用于验证状态与估计耗时，不用于精度结论。

```powershell
$env:PYTHONUTF8='1'
F:\conda\envs\torch\python.exe -m scripts.run_cuda_tests

# 使用已提交并可远程取得的完整源码 SHA；父目录须包含原始 campaign/summary 和三个父 checkpoint。
F:\conda\envs\torch\python.exe -m scripts.factorial_pipeline `
  --revision <完整源码SHA> --submit-long `
  --parent-output campaign_runs/factorial_parent_local/ferminet_results `
  --kaggle C:\Users\Administrator\AppData\Local\Programs\Python\Python313\Scripts\kaggle.exe
```

流程先执行真实双 T4 短测，核验四组更新、父状态/RNG、数值审计、观察记录和两卡日志。按实测更新、观察记录、分析和预热耗时，外推每卡六组正式实验；使用 1.25 倍安全系数，并预留 900 秒启动时间。预计墙钟时间超过 28800 秒时生成 `timing_rejected` 回执，禁止正式提交，不自动缩短科学预算。正式会话上限 32400 秒。

`schedule_seed=20261025` 固定两卡执行顺序；每卡六组，每个父种子各有两组位于每卡，同一包络的两组共享零步评估。调度和完整配置写入 campaign manifest。执行中保存检查点及失败状态，原始观察与权重不会因失败而删除。

正式任务完成后下载完整产物并生成可核验报告：

```powershell
kaggle kernels output hwbwhzbsh/ferminet-factorial-accuracy -p campaign_runs/factorial_final --page-size 100 --quiet
F:\conda\envs\torch\python.exe -m scripts.factorial_report campaign_runs/factorial_final `
  --revision <完整源码SHA> --archive physics_evidence/factorial_final_<日期>
```

收集工具核对归档 SHA、检查点、更新数、轨迹形状和重算统计，生成 `REPORT.md`、`verified_summary.json` 和原始 JSON/log 归档。大型 `.pt` 保留于下载目录；归档说明记录实际位置与哈希。

不搜索 L、学习率或网络规模，不使用中途最低能量挑组选时长或检查点。原有诊断、历史源码和历史证据保留原始内容。

全有限训练批次集中读取有效性计数，随后复用已验证张量并调用同一裁剪公式；部分非有限输入沿用原有拒绝、警告和裁剪规则。回归对照同时检查损失梯度、参数、walkers 和 RNG；性能预算由真实双 T4 实测判定。

运行中的日志可用 `kaggle kernels logs hwbwhzbsh/ferminet-factorial-accuracy --follow` 查看。需要有界采集时可使用 `KaggleApi.kernels_logs_stream` 并在读取快照后关闭流。`kernels output` 的持久化日志在运行期间可能为空；应以实时日志与任务状态分别核对进度。
