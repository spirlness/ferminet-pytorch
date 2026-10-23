# 从第 20000 轮延长训练：Kaggle 双 T4

日期：2026-10-23。本轮只改变训练时长、学习率调度和执行效率，保持模型规模、能量裁剪和正式采样步数不变。基线是[六组已归档结果](KAGGLE_T4_FINAL_RESULTS.md)；六组均从各自最终 checkpoint 开始，未按评估结果挑选种子。延长训练是否改善化学精度，必须等待新的独立冻结评估。

## 实验与学习率

配置：[长期续训](configs/kaggle_continue_t4x2.json)、[双卡 1 轮续训测试](configs/kaggle_continue_smoke.json)。固定父任务为 `hwbwhzbsh/ferminet-t4x2-multiseed`，父源码为 `d2618dfb6780f1da49f0bb4ab247d29ec15e41f8`；配置逐一固定六个最终 checkpoint 的 SHA-256，挂载输入不符会失败。

每组由 20000 轮延长到 **50000 轮**，即新增 **30000 次参数更新**。新的每组训练时间预算为 8400 秒，只计本轮新增时间，不把父任务训练耗时扣进预算；若先达到时间预算，报告真实轮数并标记 `time_budget`，不能当作完成全部计划。两卡全局预算 34200 秒，会话上限 36000 秒。

在当前已完成的第二阶段上继续，不重新冻结 Jastrow，不重置 Adam 动量，也不重新随机初始化 walkers。学习率调度从新的第 0 步开始：

```text
s = 本轮新增的参数更新数
peak_backbone = 1e-4
peak_jastrow  = 1e-5
factor(s) = 0.1 + 0.9*s/500                 （0 <= s < 500）
            max(0.1, 1/(1+(s-500)/10000))   （s >= 500）
lr(s) = peak * factor(s)
```

| 累计轮数 / 新增步数 | 主干学习率 | Jastrow 学习率 |
|---|---:|---:|
| 20000 / 0 | 1e-5 | 1e-6 |
| 20500 / 500 | 1e-4 | 1e-5 |
| 30500 / 10500 | 5e-5 | 5e-6 |
| 50000 / 30000 | 约 2.53e-5 | 约 2.53e-6 |

初始值接续旧任务的末期学习率，经过温和预热后再缓慢衰减，避免在预定轮数结束时再次快速压到很低的下限。这里是待验证的优化方案，不保证优于原余弦调度。

`VMCDriver.extend_from_checkpoint(path, expected_sha256=...)` 是显式实验迁移：只允许完成第二阶段的父状态、扩展总轮数以及修改上述学习率参数。更改 walker 数、采样步数、裁剪方式等科学配置会被拒绝。迁移信息写入新 checkpoint 和结果，保留父 checkpoint 哈希与父配置；`load_checkpoint` 仍用于相同配置下的恢复，不会静默放宽约束。`driver.run()` 无参数时只运行到配置中的累计总轮数，`driver.run(n)` 则明确新增 n 步。

## 执行效率与数值边界

- 正式全电子采样把接受数保留在 GPU，多步后一次回传；转移顺序与随机数调用顺序不变。独立测试核对坐标、缓存、累计计数及 CPU/CUDA RNG；预热和需要步长自适应的路径仍逐步执行。
- 两卡交错分配主干：GPU 0 为 FermiNet 11 → Psiformer 22 → FermiNet 33，GPU 1 为 Psiformer 11 → FermiNet 22 → Psiformer 33，减少所有较慢主干集中在同一张卡造成的尾部等待。不是 DDP，显存不合并。
- `sampling_execution="auto"` 仅考虑编译无梯度采样前向。先在三组构型上检查符号和对数波函数，然后测量整个 VMC 更新的编译/非编译 ABBA 分组耗时；只有至少快 5%、预计节约时间能覆盖编译开销才选择编译，否则使用 eager。
- 完整更新探测共执行 44 个临时更新；之后从临时完整 checkpoint 恢复所有训练状态和 RNG，丢弃探测历史，不改变正式起点。编译初始化本身也保存并恢复 CPU/CUDA RNG。选择与耗时写入每组 `sampling_performance.json` 和主日志；恢复已有续训时固定沿用已记录的执行方式。
- 短测试也按 30000 个更新的收益窗口选执行方式，使 1 轮测试实际覆盖长任务的执行候选。这个收益窗口是估计，不代表短测试执行了 30000 步。
- 不编译局域能量求导或参数梯度，不使用 AMP、TF32、随机动能估计，也不减少 10 步训练 MCMC 或独立评估样本。
- 编译可能改变 FP32 运算舍入和后续接受路径，因此不能声称与旧任务逐位连续。固定点数值检查不是全空间误差界；同环境、同执行方式的状态恢复测试不保证跨版本/硬件重现。

本地 RTX 3060 Laptop 的 40 步测量没有显示减少同步可以稳定加速全部主干；编译采样前向的测量中，FermiNet 约 1.30×，Psiformer 约 0.92×。这正是采用完整更新测速后再选择执行方式的原因。这些是开发机结果，不是 T4 加速比，见[同步路径对比](physics_evidence/continuation_local_speed_20261023.json)及[编译对比](physics_evidence/continuation_compiled_speed_20261023.json)。

## 2026-10-23 实际验收与提交

- 固定续训源码：[114377b6be860d6f4e19de1642a76ce957119a12](https://github.com/spirlness/ferminet-pytorch/commit/114377b6be860d6f4e19de1642a76ce957119a12)。后续文档与证据提交不改变这次运行源码。
- 本地完整严格 CUDA 验收：**277 通过、0 失败、0 错误、0 跳过，111.11 秒**，含真实 CUDA 及编译用例。[摘要与源码哈希](physics_evidence/continuation_cuda_final_20261023/summary.json)、[原始日志](physics_evidence/continuation_cuda_final_20261023/tests.log)。完整套件包含 CPU 检查，不表示每项均在 GPU 执行。
- 本地还实际加载两个主干的已归档 checkpoint，均从 20000 轮完成到 20001 轮；父哈希、数值审计、完整状态回滚及有限 FP64 小轨迹通过。[本地实际短测摘要](physics_evidence/continuation_local_smoke_20261023/summary.json)。
- [真实双 T4 短测](https://www.kaggle.com/code/hwbwhzbsh/ferminet-t4x2-continue-smoke) version 1 已完成。每卡 **8 项运行时检查**通过，无跳过；两个主干各净新增 **1 次更新**，最终 checkpoint/history 均为 20001 轮，父 SHA、前后数值审计、`(8,32)` FP64 小轨迹及两张卡的主日志转发核验通过。[通过回执](physics_evidence/continuation_t4_smoke_20261023/gate.json)、[实际摘要](physics_evidence/continuation_t4_smoke_20261023/ferminet_results/summary.json)、[主控制台日志](physics_evidence/continuation_t4_smoke_20261023/ferminet-t4x2-continue-smoke.log)。小轨迹的统计诊断为证据不足。
- 回执于香港时间 **12:10:50** 通过后，自动流程于 **12:10:53** 提交了[长期续训](https://www.kaggle.com/code/hwbwhzbsh/ferminet-t4x2-continue-multiseed) version 1。**12:12:56** 的实际日志已确认两卡均训练中，FermiNet seed 11 为 20063 轮，Psiformer seed 11 为 20001 轮；其余种子等待同卡顺序执行。[启动快照](physics_evidence/continuation_submission_20261023.json)、[原始启动日志](physics_evidence/continuation_long_start_20261023.log)。这是启动证据，尚未完成新的长期冻结评估。

长期任务启动时的完整更新测速如下；两组均通过数值检查并选择 compiled。它们是短时、同一会话中的测速，不能保证全程或后续种子具有相同加速比。短测会话曾对 FermiNet 选择 eager，说明执行方式选择会受测量时负载影响，未强制全卡编译。

| 主干 | eager 秒/更新 | compiled 秒/更新 | 测得吞吐比 |
|---|---:|---:|---:|
| FermiNet | 0.1972 | 0.1613 | 约 **1.22×** |
| Psiformer | 0.2520 | 0.2011 | 约 **1.25×** |

测速的临时 44 次更新均已丢弃并恢复起点；正式长期训练仍从各自第 20000 轮开始。没有把测速能量用于选择 checkpoint 或报告精度。

## 测试后自动提交

先提交并推送源码，使用完整 SHA。以下命令运行于已配置 CUDA PyTorch 的 Python 环境，`--kaggle` 指向本机已认证的 Kaggle CLI，无需在该 Python 环境额外安装 Kaggle SDK：

```powershell
$env:PYTHONUTF8='1'
F:\conda\envs\torch\python.exe -m scripts.run_cuda_tests

F:\conda\envs\torch\python.exe -m scripts.kaggle_pipeline `
  --owner hwbwhzbsh --revision <完整源码SHA> --continuation --submit-long `
  --kaggle C:\Users\Administrator\AppData\Local\Programs\Python\Python313\Scripts\kaggle.exe
```

流程依次生成私有双 T4 测试、提交、等待执行完成、下载真实产物、验证父检查点和净新增 1 次更新及两卡日志、生成回执，最后才提交同一源码和配置的长期任务。未通过测试不会提交长期任务。去掉 `--submit-long` 只进行短测；重复执行会通过 `.kaggle_deploy/pipeline/<SHA前12位>/pipeline.json` 接续已成功阶段，通常无需重新提交。日志与产物验证需要时间，队列等待不代表开始训练。

此流程调用原 `VMCDriver.step`，没有第二套正式训练更新核心。临时测速也复用该核心。短流程仅证明迁移和执行链路，评估样本仅 `(8,32)`，不能用于化学精度结论。每张 T4 还会执行既有运行时回归和新增续训回归，任何失败阻止实验启动。

## 独立评估

长期计划仍为每组最后 checkpoint 的 FP64 独立评估：512 walkers × 2048 条记录，每条推进 20 步，独立预热 1000 步，不裁剪能量，保留时间/链结构。新评估种子为 FermiNet 400011/400022/400033、Psiformer 500011/500022/500033，与父实验不同。

报告全部六组的偏差、SEM、混合诊断和数值审计；化学精度仍要求 `abs(E-E_ref)+1.96*SEM <= 0.0015936 Ha`。不以训练最低能量选择 checkpoint，不把临时测速或 1 轮短测当作精度证据。
