> 历史记录：0.2.0 的重构与测试结果。当前 0.3.0 接口及验证见 [第二轮报告](REFACTOR_PHASE2_REPORT.md)。

# 第一轮简化重构与验收

日期：2026-10-22。版本：0.2.0。此轮按用户要求不维护旧接口，范围为统一包络、清理原型测试和规范包结构。

## 实际变化

- 运行库集中到 `ferminet_pytorch/`，测试集中到 `tests/`，基准和诊断集中到 `scripts/`，演示位于 `examples/`。
- 删除根目录运行模块、重复根 `__init__.py`、editable namespace shim、`__path__` 扩展及包内导入回退。运行代码不再修改 `sys.path`。
- 仅保留 `ferminet_pytorch.envelope.NuclearEnvelope`。输入为 `(B,N,M)` 距离，输出为 `(B,K,N,O)`。几何计算由模型和几何模块完成。两个模型都使用相同的指数/Gaussian 包络、参数定义和初始化。
- 删除 `envelope_modern.py`、整个 `legacy_adapters.py` 和对应导出。没有新增旧接口包装器。旧的坐标包络调用会失效，应直接迁移调用者。
- 删除三份内嵌实现的原型测试文件和旧版包络对照文件；不再测试复制在测试文件内部的网络、动能计算器或适配器。
- 新包络验证采用显式标量数学求和、径向解析极限，并比较坐标及参数的一、二阶导数；新增 CUDA 上相同检查。有效自旋池化与 attention 检查迁移到正式模块。
- wheel 直接打包 `ferminet_pytorch/`，不再维护重复的 Python 文件清单；sdist 保留测试、示例和脚本。新增 `ferminet-train`、`ferminet-validate` 控制台入口。

本次没有宣称消除包络的五维广播中间量，也没有统一两种模型、交互层及训练核心；这些属于后续阶段。

## 代码规模

按本轮开始时的工作区统计，包含注释和文档字符串：

| 口径 | 重构前 | 重构后 | 减少 |
|---|---:|---:|---:|
| 运行库，包含验证入口和两份原包导出 | 5115 行 | 4366 行 | 749 行，约 14.6% |
| 测试 | 9709 行 | 7617 行 | 2092 行，约 21.5% |

运行库口径将原来顶层的 `physics_validation.py` 和 namespace shim 纳入统计，避免目录移动造成不一致比较。历史 `physics_evidence/20261022T045238Z/source_snapshot` 不属于当前运行库，不修改历史快照以制造代码减少。

## 测试迁移

原 270 项中，三份原型文件共 26 项、旧包络对照 2 项、适配器专用 3 项退出当前套件。新增 8 项正式实现检查及 4 项 CUDA 包络检查，当前为 251 项。数量变化不被当作覆盖率提高的证据。

| 原检查职责 | 当前验证 |
|---|---|
| 几何形状、单电子边界 | `tests/test_geometry_cache.py` |
| 排除自身的自旋池化、开壳层、二阶导数 | `test_production_contracts.py`、`test_layers_modern.py` 和已有电子自相互作用回归 |
| 包络形状、核电荷初始化与数值 | `test_envelope.py`，含独立标量求和和坐标/参数导数；`test_cuda.TestCudaEnvelope` |
| 谐振子动能与三后端 | `test_kinetic_modern.py`、`test_physical_matrix.py`，后者含非对角 Hessian 的 Hutchinson 方差验证 |
| 自旋区分、同自旋置换、单电子 attention | `test_production_contracts.py`、`test_psiformer_integration.py` |
| 完整波函数尖峰、反对称性、自旋和奇异导数 | 保留上一轮物理回归及 CUDA 数值检查 |
| 旧适配器签名、旧原型内部缺陷和兼容行为 | 随已删除接口退出；不再维持这些期望 |

移除的测试名称保存在 [清单](physics_evidence/phase1_removed_test_inventory.json)，职责映射见 [迁移记录](physics_evidence/phase1_test_migration.json)。独立数学参照被保留；没有把所有参照替换成对正式实现自身的比较。

## 实际验收

- CUDA 可用环境完整套件：**251 通过、0 失败、0 错误、0 跳过**，81.78 秒。RTX 3060 Laptop GPU / PyTorch 2.14.0 / CUDA runtime 13.0。
- 完整套件包含真正执行 CUDA 的数值、梯度、包络、自旋、奇异矩阵及 Inductor 检查，也包含 CPU 测试；并非所有 251 项都在 GPU 上执行。
- CPU 独立环境：包络、迁移后的正式模块契约及两组物理审计测试，共 26 项通过。
- `uv sync` 更新 editable 安装至 0.2.0；sdist/wheel 构建及 checkout 外的隔离导入通过。分发测试导入包内每个运行模块，并确认 wheel 不包含已删除模块或开发脚本。
- 新控制台入口执行 CPU 单轮训练成功；`python -m examples.example_h2` 演示成功。
- 九个历史 checkpoint 与原始源码快照逐组回放，符号一致、对数波函数最大差为 0、局域能量最大差为 `3.56e-15 Ha`（向上取整）。参数可用于此次审计回放不构成对旧接口或所有历史 checkpoint 的兼容承诺。

首轮新套件有两处 mock 使用旧顶层 `evaluation` 路径而报错；已改为正式包路径，原始失败记录保留。修复后的完整套件没有放宽数值容限。

证据：

- [完整测试摘要及源码哈希](physics_evidence/phase1_cuda_20261022T053924Z/summary.json)
- [完整测试日志](physics_evidence/phase1_cuda_20261022T053924Z/tests.log)
- [首轮失败日志](physics_evidence/phase1_cuda_20261022T053617Z/tests.log)
- [九组 checkpoint 回放](physics_evidence/20261022T045238Z/phase1_code_equivalence.json)
- [训练命令验证](physics_evidence/phase1_cli_smoke.log)
- [演示验证](physics_evidence/phase1_example_smoke.log)

## 新入口

```powershell
uv sync --locked
uv run python -m unittest discover -s tests -t . -v
F:\conda\envs\torch\python.exe -m scripts.run_cuda_tests
uv run ferminet-train --epochs 5 --walkers 64
uv run python -m examples.example_h2
uv build
```

严格 CUDA 入口每次创建新证据目录，CUDA 不可用或任何测试跳过均失败，编译器环境调整仅限子进程。历史训练记录、快照及上一轮验收结果没有重写为本次结果。本次验收不提供新的化学精度或吞吐提升结论。
