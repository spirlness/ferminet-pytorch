# 开发指南（0.3.0）

运行库使用一个交互层、一套波函数组装和一个优化核心。当前入口和验收记录见 [README](README.md) 与 [第二轮重构报告](REFACTOR_PHASE2_REPORT.md)。旧的 Modern/Composable 模型模块、四组件包装和多关键字训练接口已删除。

完整包职责和旧接口替换表见[重构架构与迁移说明](ARCHITECTURE_REDUNDANCY_REVIEW_AND_REFACTORING_PROPOSAL.md)。

## 数据流

```text
MCMCSampler.walkers -> model geometry/features -> selected interactions
                  -> spin orbitals * nuclear envelopes
                  -> signed determinant sum -> Jastrow -> spin projection
                  -> independent coordinate energy / parameter surrogate graphs
                  -> VMCDriver update -> sampler cache refresh
```

`MCMCSampler` 独占 walkers 和当前对数波函数缓存。Driver 不维护第二份 walkers，也不持有跨坐标求导图的几何缓存。

## 交互层

```python
from ferminet_pytorch.layers import FermiInteractionLayer
layer = FermiInteractionLayer(8, 4, 32, 8, n_up=2, n_down=1)
h1_next, h2_next = layer(h1, h2)
```

自旋数在构造时固定；改变电子数需要创建新层。单电子更新仍是 `[h_i, mean_up, mean_down, pair_mean_up, pair_mean_down]` 的线性变换加残差。权重存储为一个线性矩阵，在前向中拆分应用，避免创建广播拼接张量。池化矩阵在构造时建立，对角线权重为零，计数为零的通道归零；没有原地扣除对角线造成的大数抵消问题。

这些操作仍产生中间结果，静态池化矩阵也占用空间；本次不提供零分配或固定加速比保证。

## 波函数

```python
from ferminet_pytorch import FermiNet
from ferminet_pytorch.systems import build_h2_system
system = build_h2_system()
model = FermiNet(system.nuclear_coords, system.nuclear_charges, 1, 1,
                interaction_type="psiformer", use_cusp_jastrow=True,
                spin_adapted=True)
log_abs, sign = model.forward_log_psi(r)
psi = model(r)
```

`interaction_type` 可为 `ferminet` 或 `psiformer`，未知值报错。两者共用几何规则、轨道、包络、行列式、Jastrow 和自旋投影。Psiformer 的注意力块属于不同物理网络主干，并非需要删除的重复实现。

`forward_log_psi` 始终返回 `(log_abs, sign)`，`forward` 返回波函数标量。尖峰参数及包络根据实际 Jastrow 与投影自旋校验；不兼容配置直接报错。

投影会重排并扩展坐标，必须为投影构型重建几何。传入缓存必须属于相同坐标对象、平滑规则和坐标梯度图。几何上下文的冻结 dataclass 不防止 tensor 原地修改，调用方应将坐标按只读方式处理。

## 统一包络

```python
from ferminet_pytorch.envelope import NuclearEnvelope
# distances: (B,N,M), result: (B,K,N,O)
envelope = NuclearEnvelope(4, 2, 1, envelope_type="gaussian")
result = envelope(distances)
```

核坐标到距离的计算由模型与几何模块负责。包络仍包含五维广播中间量。若引入依赖方向位移的各向异性包络，须显式修改轨道调用并重新核验尖峰，不能直接替换当前标量距离接口。

## 统一训练

```python
from ferminet_pytorch import TrainingConfig, train_vmc
config = TrainingConfig(num_epochs=20, num_walkers=128, device="cpu",
                        use_cusp_jastrow=True, spin_adapted=True,
                        two_stage_jastrow=True, jastrow_stage1_epochs=10,
                        jastrow_lr=1e-4)
result = train_vmc(config=config)
```

也可提供显式的 `system` 和 `model`；此时模型结构由传入模型决定。配置中的 `use_cusp_jastrow`、`interaction_type` 等模型选项只用于创建默认模型。它们不会在训练中替换已提供的模块。

直接构造 Driver 可控制采样器、优化器、调度器和动能后端：

```python
from ferminet_pytorch import VMCDriver
from ferminet_pytorch.mcmc import MCMCSampler
sampler = MCMCSampler(model, system, num_walkers=128)
sampler.burn_in(50)
driver = VMCDriver(model, system, sampler, config=config)
driver.run(20)
```

直接使用时，由调用方把模型和体系移到所需设备；采样器的默认精度跟随模型。Driver 要求采样器共享同一模型与体系实例。

每步只在 `VMCDriver.step` 中执行以下逻辑：冻结提议自适应的采样、独立坐标图局域能量、脱图后参数代理损失、更新前最佳状态快照、梯度裁剪、优化、刷新采样缓存和调度。`VMCLoss.compute_step_loss` 是共同的图隔离边界，局域能量不回连参数损失。

两阶段训练由同一个 Driver 切换，第一阶段冻结 Jastrow，第二阶段解冻并为其分配独立学习率。若请求恢复第一阶段最佳状态，先恢复并刷新缓存再开始第二阶段。只对有限训练能量更新最佳记录。

单步训练的最低批次能量不是经过独立评估的精度证明。最终 checkpoint 可选择恢复最佳状态；应另外调用冻结评估。保存文件同时区分更新后模型与更新前测得能量对应的最佳模型，优化器状态属于更新后模型。

`driver.run(n)` 在当前状态上再执行 n 步。`driver.restore(state_dict)` 仅恢复模型权重并刷新缓存。Kaggle 扩展新增 `load_checkpoint(path)`：在新建 Driver 上恢复 schema-2 的默认 Adam/余弦调度、训练阶段、walkers、采样计数和 Torch CPU/当前 CUDA 设备随机状态，要求相同配置、dtype 和 CPU/CUDA 后端；同环境一致性已由专门测试覆盖，跨版本或硬件不保证逐位重现。详见 [Kaggle 训练指南](KAGGLE_T4_TRAINING_GUIDE.md)。`train_vmc` 返回的 `final_energy` 是最后一个训练批次更新前的能量，可能与恢复后的模型不同；独立评估使用 `eval_result`，且需要通过混合诊断。

`TrainingConfig` 默认 `clip_method="tanh"`；截断会改变梯度目标，不能称为严格无偏估计。使用 `clip_method="none"` 关闭能量截断。默认 `grad_clip=1.0`，设为 0 关闭梯度裁剪。

完成训练后的学习率迁移使用 `extend_from_checkpoint(path, expected_sha256=...)`，而非放宽 `load_checkpoint` 的配置检查。新 `lr_schedule="warmup_inverse"` 支持温和预热和缓慢末期衰减；Adam 状态、walkers、缓存及随机状态继续保留，新的调度从迁移起点计数。详见[续训与开发工作流](CONTINUATION_TRAINING_GUIDE.md)。`driver.run()` 无参数只执行到累计目标轮数，显式 `run(n)` 在当前状态上新增 n 步。

## 验证与证据

```powershell
uv run python -m unittest discover -s tests -t . -v
F:\conda\envs\torch\python.exe -m scripts.run_cuda_tests
uv build
```

CPU 环境会跳过 GPU 检查；严格 CUDA 入口拒绝 CUDA 不可用及任何跳过项。数值、物理、编译检查与长期训练收敛分开报告。

独立数学参照包括显式逐电子求和、解析氢原子与谐振子、行列式多项式。离线审计脚本 `scripts.verify_phase2_refactor` 使用保存的第一轮源码，显式映射权重并比较旧模型和当前模型；这些转换仅用于审计，不属于运行库兼容接口。
