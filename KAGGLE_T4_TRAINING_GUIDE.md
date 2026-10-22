# Kaggle 双 T4：多种子长期训练与独立冻结评估

日期：2026-10-22。训练和评估任务自动执行；提交成功不代表已经完成训练或达到化学精度。当前源码通过 GitHub 的完整 commit SHA 拉取，Kaggle notebook 默认为私有。

## 实验计划

配置文件：[长期实验](configs/kaggle_t4x2.json)、[双卡短流程](configs/kaggle_smoke.json)。体系固定为 H₂、核间距 1.4 bohr、单重态；参考 BO 能量为 −1.174475714220 Ha。

| 项目 | 长期配置 |
|---|---|
| 模型和种子 | FermiNet、Psiformer，各 11/22/33，共 6 个实验 |
| GPU 分工 | 一张卡一个进程，独立参数与采样链；每卡顺序执行 3 个种子 |
| 模型规模 | 8 行列式、3 层、单/双电子维度 64/16；Gaussian 包络、Kato Jastrow、自旋投影 |
| 训练 | FP32、512 walkers、每轮 10 步 MCMC；最多 20000 轮或 8400 秒 |
| 优化 | Adam 初始学习率 1e-3；前 1000 轮冻结 Jastrow，再解冻并使用 1e-4 学习率；余弦调度、梯度范数裁剪 1 |
| 能量截断 | 训练默认 tanh，倍数 5；改变梯度目标，不是严格无偏估计；冻结评估不截断能量 |
| 训练采样 | 独立预热 500 步；参数更新后刷新缓存；训练测量时不调整提议步长 |
| 评估 | 将最后 checkpoint 的权重精确提升到 FP64；独立随机种子、新采样器、新预热 1000 步 |
| 评估采样 | 512 walkers，目标 2048 条时间记录；每条之前推进 20 步，生产期冻结自适应；最多 1800 秒 |
| 评估显存 | 每次局域能量只计算 128 walkers，仍保留完整 `(time, walker)` 轨迹 |
| 保存 | 每 300 秒原子保存 checkpoint，训练结束再保存最终 checkpoint、历史及评估轨迹 |

每张卡三个实验的预算约 8.5 小时，两卡合计约 17 GPU 小时，另加环境、预热和数值检查开销。Kaggle 限额以账号实时返回为准；提交时使用 10 小时会话上限，预留结束保存时间。短流程设置极少训练与采样，仅验执行链路，统计诊断应报告证据不足。

两种主干共用 `VMCDriver.step`；没有第二套优化器更新循环。最终 checkpoint 按预先声明的训练结束条件选择，不按训练最低能量或独立评估结果挑选。若时间预算先耗尽，保留实际轮数及 `time_budget` 状态；这些结果不能直接用于架构速度或精度排名。

## T4 配置选择

- 两卡运行独立种子，提高独立实验吞吐；小型 H₂ 网络使用这种方式避免每轮梯度通信，两卡显存不合并。
- 训练保留完整 FP32，不使用 FP16/BF16 autocast；坐标 Hessian、节点附近和尖峰抵消对数值误差敏感。禁用 TF32，选择精确反向动能后端；本轮不使用随机 Hutchinson 迹估计。
- 冻结评估使用 FP64 和小批次；精度提升不改变存储的参数值，但浮点求值与 FP32 有区别，另行记录精度检查。
- 每个进程只看见一张 GPU，CPU 线程设为 1，减少共享 CPU 资源争用。保留 Kaggle 自带 CUDA PyTorch，不下载替换大体积 torch wheel。
- 默认不启用 `torch.compile`，避免把未知编译成本计入长实验；需要加速比时另做受控基准。

训练前后均在指定采样构型上比较同权重的 FP32/FP64 对数波函数、符号和局域能量。预检查不通过则该实验失败；结束检查不通过仍保存 FP64 评估，但禁止化学精度认证。固定点检查不是全空间误差上界。

## 推送与运行

先将工作区提交并推到 GitHub，确保完整 SHA 在远端可获取。以下命令在仓库根目录执行，Kaggle 凭据保留在本地用户配置中，不写入仓库或 notebook：

```powershell
# 本机完整 CUDA 验收，任何 skip 都失败
F:\conda\envs\torch\python.exe -m scripts.run_cuda_tests

# 生成当前 HEAD 对应的短流程提交
F:\conda\envs\torch\python.exe -m scripts.prepare_kaggle --owner hwbwhzbsh --mode smoke --output .kaggle_deploy/smoke
kaggle kernels push -p .kaggle_deploy/smoke --accelerator NvidiaTeslaT4 -t 1200
kaggle kernels status hwbwhzbsh/ferminet-t4x2-smoke

# 短流程确认两张 T4 均执行成功后，再启动长期实验
F:\conda\envs\torch\python.exe -m scripts.prepare_kaggle --owner hwbwhzbsh --mode long --output .kaggle_deploy/long
kaggle kernels push -p .kaggle_deploy/long --accelerator NvidiaTeslaT4 -t 36000
kaggle kernels status hwbwhzbsh/ferminet-t4x2-multiseed
kaggle kernels output hwbwhzbsh/ferminet-t4x2-multiseed -p campaign_runs/kaggle_download
```

Bootstrap 校验实际 GPU 必须恰好为两张 T4，没有 CPU 回退。分别在两卡执行 CUDA 断点恢复/分批评估测试，再启动两个实验进程。它记录源码 SHA、包版本和 GPU 名称，并校验实际 checkout 与提交一致。

官方提交格式和 T4×2 标识见 [Kaggle CLI kernels 文档](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md)及[元数据规范](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels_metadata.md)。

本机最终完整 CUDA 验收：261 通过、0 失败、0 错误、0 跳过，94.64 秒，RTX 3060 Laptop GPU / PyTorch 2.14.0 / CUDA runtime 13.0。[原始日志](physics_evidence/kaggle_final_cuda/tests.log)与[源码哈希及结果](physics_evidence/kaggle_final_cuda/summary.json)已保留。该记录不能代替 Kaggle 实际双 T4 验收或长期收敛结果。

## 输出与判读

所有实验输出在 `/kaggle/working/ferminet_results/`：

```text
bootstrap.json, campaign.json    软件/硬件、配置、源码版本及哈希
runtime_tests_gpu0/1.log          两卡的运行时检查
worker_0/1.log                   分别记录两个进程的训练输出
summary.json                     全部计划种子、完成/失败状态和跨种子摘要
ferminet_seed11/ ...             每种主干和种子的独立目录
    manifest.json                必须匹配的源码与实验计划
    resume.pt                    更新后模型、Adam、调度、阶段、walkers、Torch CPU/CUDA RNG
    final_checkpoint.pt          预先声明的最终训练状态
    training_history.json        未用于挑选 checkpoint 的训练历史
    precision_before/after.json  指定采样点的精度对照
    evaluation_trajectory.pt     未截断的 FP64 局域能量，保留时间/链结构
    result.json                  能量、SEM、区间、ESS、Rhat、分块、参考和失败原因
```

单次评估要求至少 128 条时间记录、ESS ≥1000、能量 split-Rhat ≤1.01 和分块平台。满足这些诊断且 `abs(mean-reference) + 1.96*SEM ≤0.0015936 Ha`，并通过数值精度检查后，才标记达到化学精度。任何非有限评估能量都导致失败，不能丢弃后继续报告精度。

跨种子摘要保留所有计划项；失败、缺失或诊断不足不能当作成功。三次完整训练会报告 checkpoint 能量均值及考虑种子间变化的 Student t 区间，它不是把不同波函数的样本拼成一条轨迹。单 checkpoint 的 Monte Carlo SEM 不包含优化误差。

## 断点继续

同一 checkout、同一配置、同一输出目录再次运行时，会跳过已完成实验，并从 `resume.pt` 继续未完成训练。schema-2 续训恢复默认 Adam/余弦调度、Jastrow 阶段、最佳记录、采样计数与 Torch CPU/当前 CUDA 设备随机状态，恢复后刷新缓存；必须在相同 dtype、CPU/CUDA 后端下创建新的 Driver。CPU 与 CUDA 回归分别验证同一软件/硬件环境内的后续参数、walkers 和随机状态一致，不保证跨硬件或 PyTorch 版本逐位相同。

Kaggle 新会话需要将前一次输出作为 kernel source 挂载，并使用原来的完整源码 SHA：

```powershell
F:\conda\envs\torch\python.exe -m scripts.prepare_kaggle --owner hwbwhzbsh --mode long --revision <原完整SHA> --resume-kernel hwbwhzbsh/ferminet-t4x2-multiseed --output .kaggle_deploy/resume
kaggle kernels push -p .kaggle_deploy/resume --accelerator NvidiaTeslaT4 -t 36000
```

Bootstrap 只接受唯一、源码 SHA 匹配的挂载实验，再由各实验校验完整配置和源码哈希。恢复不重新预热已经保存的训练链；独立冻结评估仍使用新采样器。会话中断可能损失最近一个保存周期的进度，必须先确认上次输出可供挂载及剩余额度。
