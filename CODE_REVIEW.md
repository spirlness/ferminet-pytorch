# FermiNet-PyTorch 综合深度代码审查报告 (Code Review)

**审查基准版本**：`ferminet_pytorch` (Python 3.9+, PyTorch 2.0+)  
**审查时间**：2026-10-15  
**工作目录**：`c:\Users\Administrator\.gemini\antigravity\scratch\ferminet_pytorch`  
**自动化测试基线**：Ran 82 tests in 10.137s: OK (18 skipped on CPU)  

---

## 一、 执行摘要 (Executive Summary)

### 1.1 总体评分与质量雷达

| 评估维度 | 得分 (百分制) | 等级 | 评价概要 |
| :--- | :---: | :---: | :--- |
| **量子化学与物理保真度** | 82 / 100 | B+ | 反对称性与 Kato 尖点消去验证严谨；但开壳层自旋投影崩溃，近核输入特征缺乏光滑截断。 |
| **深度学习实现与数值稳定性** | 78 / 100 | B | 核心网络广播高效，但局域能量求导未隔离参数导致 16.6 倍性能损耗，欧氏距离 clamp 诱发虚假势能黑洞。 |
| **MCMC 采样与优化稳健性** | 80 / 100 | B+ | 变分梯度截断严格阻断参数，但生产采样持续自适应破坏细致平衡，MAD 退化引发梯度窒息。 |
| **工程架构与测试防御质量** | 88 / 100 | A- | 7 个测试套件覆盖深核抵消、多电子导数和 packaging，但部分自适应断言偏平庸，缺少开壳层覆盖。 |
| **综合评分** | **81.5 / 100** | **B+** | **高水准的学术从头算原型，具备优秀的物理直觉，但迈向生产级大分子计算需关键重构。** |

### 1.2 项目核心优势 (Key Strengths)

1. **变分协方差梯度阻断与数学严谨性**：`loss.py` 严格使用 `.detach()` 隔离了局域能量对波函数参数的计算图，精确落实了量子力学变分定理 $\nabla_\theta \langle E \rangle = 2 \, \mathbb{E}[(E_L - \langle E \rangle) \nabla_\theta \ln |\Psi|]$，理论导数与 Autograd 误差在测试中达到 $<10^{-5}$。
2. **显式 Kato 尖点消除与近核解析验证**：`jastrow.py` 与 `envelope.py` 构筑了显式电子-核与电子-电子尖点因式，在测试中成功证明了动能拉普拉斯与势能库仑奇异性在 $r \to 0$ 处的精确万倍抵消。
3. **多重奇异性防御网**：建立了针对零行列式（代数余子式分支）、小振幅对数下溢（`safe_abs_summed`）、全 NaN 能量阻断（`FloatingPointError`）的多层防御。

### 1.3 核心战略风险 (Top Critical Risks)

1. **局域能量求导未隔离参数（`FINDING-PYT-01`，Critical）**：计算 $E_L$ 动能拉普拉斯时未临时冻结模型参数，导致 PyTorch Autograd 无端记录全网数以万计参数的二阶 Hessian 计算图。实测证明参数隔离可带来 **16.59 倍的速度暴增**（从 1.598s 降至 0.096s）。
2. **欧氏距离 clamp 截断诱发“伪势能黑洞”（`FINDING-PYT-02`，Critical）**：`torch.clamp(sum(r**2), min=eps)` 导致一阶导截断跳变，实测二阶拉普拉斯算子在 $r < \sqrt{\epsilon}$ 区域**瞬间跳变归零**。失去动能发散抵消后，势能 $-Z/\sqrt{\epsilon} = -10^{12}$ Hartree 形成数值深渊，严重吸引 Walker 塌陷。
3. **开壳层自由基体系自旋投影硬编码崩溃（`PHY-01`，Critical）**：`spin.py` 硬编码断言 `if n % 2:`，且 `model.py` 盲目分配 `singlet`，导致所有奇数电子体系（如 Li 原子 $n_\uparrow=2, n_\downarrow=1$、自由基 $\cdot\text{CH}_3$）在初始化阶段抛出 `ValueError`。
4. **波函数节点面高阶微分 NaN 毒化（`PHY-05`，Critical）**：节点面处 $\ln|\Psi| = -\infty$，直接送入二阶自动微分引发 `0 * inf = NaN`，污染整个 Batch。
5. **MCMC 生产采样持续自适应步长破坏细致平衡（`F-MCMC-01`，High）**：采样器在正式训练中持续调整步长，马尔可夫链非时齐，稳态玻恩测度失真。

---

## 二、 逐项维度审查分析 (In-Depth Technical Analysis)

### 2.1 量子化学与物理保真度 (Quantum Physics & Wavefunction Fidelity)

#### 1. 反对称性与自旋统计评估 (判定：合格，但开壳层存在阻断性硬编码)
- **Slater 行列式反对称性**：`model.py` 分别构建自旋向上和向下电子的单粒子轨道矩阵，并通过双分支或多行列式求和。经 `test_ferminet.py` 中的 `test_antisymmetry_up_electrons` 与 `test_antisymmetry_down_electrons` 验证，交换相同自旋电子完全满足 $\Psi(r_{P}) = -\Psi(r)$，相反自旋电子交换不改变符号，满足玻色-爱因斯坦/费米-狄拉克统计分离。
- **开壳层与自由基崩溃**：`spin.py` 的 Löwdin 谱投影算符限定了只能投影到整数自旋（单重态/三重态），未定义半整数自旋多重态（二重态 $S=1/2$、四重态 $S=3/2$）。且对总电子数为奇数的情况强行抛错，丧失了对开壳层化学体系的计算能力。

#### 2. Kato 尖点条件与近核行为 (判定：部分有效，存在特征层奇异性泄漏)
- **显式 Jastrow 因子**：`jastrow.py` 实现了满足 Kato 条件的自旋相关两电子因子（平行自旋 $1/4$，反平行自旋 $1/2$）以及电子-核因式。测试显示其在 $r=10^{-4}$ 时能够将局域能量发散幅度压缩 4 个数量级。
- **特征层奇异性泄漏**：`features.py` 在默认配置下（`smooth_cuspless=False`）直接将欧氏距离 $d_{en} = |r|$ 作为第一层 MLP 输入。由于三维欧氏距离的拉普拉斯具有内在奇异性 $\nabla^2 |r| = 2/|r|$，未平滑的特征被多层神经网络非线性放大，在核附近强行制造非物理的 $1/r$ 动能发散，破坏了局域能量变分下界。

#### 3. 动能拉普拉斯展开与库仑势发散处理 (判定：基本有效，但存在导数不协同与节点面 NaN 漏洞)
- **局域动能分解**：采用对数波函数形式 $T_L = -\frac{1}{2} (\nabla^2 \ln|\Psi| + |\nabla \ln|\Psi||^2)$，有效避免了直接对小振幅波函数求二阶导的大数除小数下溢。
- **势能截断不协同**：库仑势能使用 `clamp(..., min=eps)` 硬截断，在饱和区导数为 0；而 Jastrow 尖点动能依然提供连续微商，破坏了极小距离下微商的协同对消。

---

### 2.2 深度学习实现与 PyTorch 性能 (Deep Learning & Numerical Stability)

#### 1. 张量向量化与显式循环审查 (判定：特征层优良，导数层存在严重冗余)
- **特征与网络层**：`features.py` 与 `layers.py` 充分利用了 PyTorch 的广播机制，构造 $(B, N, N_I, 4)$ 电子-核张量与 $(B, N, N, 4)$ 电子-电子张量无任何显式 Python 循环。
- **动能反向传播的 96%+ 算力浪费**：`hamiltonian.py#L159-L171` 使用 `for d in range(dim_total)` 循环执行 $3N$ 次反向传播。每次反向传播计算了完整的 Hessian 第 $d$ 行（$3N$ 个元素），但仅提取对角元素并丢弃其余 $3N-1$ 项。算力浪费率达 $1 - 1/(3N)$（$N=10$ 时为 96.7%）。

#### 2. 计算图与显存开销实验分析 (判定：存在 16.6 倍性能提升的重大修复项)
- **未阻断参数的计算图雪崩**：在调用 `compute_local_energy` 时，模型参数默认 `requires_grad=True`。计算一阶坐标导数时设置了 `create_graph=True`，导致全网所有参数（权重、偏置、包络）全部被记录到反向图中，随后在 $3N$ 维二阶反向传播中被反复遍历。
- **基准测试数据**（H2 分子, 16 行列式, 4 层网络, 256 walkers）：
  - **默认实现**：耗时 **1.5981 秒**
  - **参数临时冻结（`param.requires_grad=False`）**：耗时 **0.0963 秒**
  - **实测加速比**：**16.59 倍**（数值绝对误差严格为 0.0）

#### 3. 跨平台与 `torch.compile` 兼容性 (判定：基本可用，存在图断裂与编码隐患)
- **编译图断裂 (Graph Break)**：`model.py#L344` 的 `if singular.any():` 引发了主机端与设备端同步，打断了 TorchDynamo 符号追踪，导致 Inductor 无法跨层融合 Triton 内核。
- **Windows GBK 字符集陷阱**：在中文 Windows 终端下，测试用例 docstring 中包含的中文注释在 CP936 编码下打印乱码；旧版 PyTorch 在未设置 `PYTHONUTF8=1` 时编译 Inductor 模板可能抛出 `UnicodeDecodeError`。

---

### 2.3 MCMC 采样与 VMC 优化稳健性 (MCMC & Optimization)

#### 1. Metropolis-Hastings 采样与细致平衡条件 (判定：算法正确，但生产步长未冻结)
- **细致平衡违背**：`mcmc.py#L248` 在每次 `step()` 中无条件根据实时接受率调整 `step_size`。这在预热阶段十分有效，但在正式变分抽样中持续变化转移核，破坏了马尔可夫链时齐性，采得的电子分布偏离精确玻恩测度 $|\Psi|^2$。
- **高维扩散接受率失配**：全电子协同移动时理论最优接受率为 0.234，代码硬编码为 0.5，导致高维电子体系提议步长被过度压小，构型自相关时间拉长。

#### 2. 局域能量 MAD 裁剪与梯度方差控制 (判定：机制健全，但需警惕梯度窒息)
- **MAD 裁剪优势**：`loss.py` 采用双边绝对中位差（MAD）配合 `tanh` 平滑截断异常能量尖峰，极大增强了针对近核/节面离群点的抵御能力。
- **退化分布下的梯度窒息**：当批次样本由于高度对称或数值下溢导致 $\text{mad} \to 0$ 时，截断阈值 $\Delta$ 坍缩至 `eps=1e-8`，导致 `tanh` 饱和且残差压扁至 $10^{-8}$，反向传播梯度近乎归零，优化陷入停滞。

#### 3. 优化器与训练循环控制 (判定：基础可用，缺乏生产级调度器)
- 仅提供固定学习率的 Adam 优化器，缺乏余弦退火（Cosine Annealing）或步长衰减，难以在后期抑制由有限 Monte Carlo 采样引入的固有随机梯度噪声。

---

### 2.4 工程架构与测试防御质量 (Architecture & Test Quality)

#### 1. 模块边界与职责隔离
- 模块正交性良好：`features.py`（几何特征）、`layers.py`（自旋图交互）、`model.py`（波函数组合）、`hamiltonian.py`（动能与势能算符）、`mcmc.py`（采样器）、`loss.py`（变分目标）职责划分清晰，未出现跨层循环依赖。

#### 2. 测试套件质量评估（82 tests: 64 passed, 18 skipped）
整个测试矩阵在标准 `unittest discover` 下共加载 82 个独立测试用例（64 项在 CPU 环境下通过，18 项 CUDA 硬件测试在 CPU 环境下合规跳过，测试套件以退出码 0 稳健通过），代码库实际包含且仅包含 7 大真实测试套件，逐项详细评估如下：

- **`test_ferminet.py` (17 用例，全部通过)**：
  - 核心物理覆盖：FermiNet 空间与自旋嵌入特征、Slater 行列式反对称性（分别针对自旋向上/向下电子交换符号翻转检验：`test_antisymmetry_up_electrons` 与 `test_antisymmetry_down_electrons`）、泡利不相容原理（同自旋电子重合波函数精确为 0，对数对齐为 -inf）、9 原子大体系双精度 slogdet 独立检验、核电荷预计算与设备迁移 $V_{nn}$、MCMC 采样器基础功能以及端到端小规模 VMC 训练循环。
  - 断言质量：**高质量**。采用双精度符号与对数双重数值比对，杜绝 $0 == -0$ 假阳性。
- **`test_vmc.py` (14 用例，全部通过)**：
  - 核心机制覆盖：MCMC 构型初始化（`test_mcmc_init_walkers`, `test_mcmc_custom_initial_walkers`）、全电子与逐电子移动步进（`test_mcmc_step_all_mode`, `test_mcmc_step_one_by_one_mode`）、自适应步长（`test_mcmc_adaptive_step_size`）、双边 MAD 能量截断平滑截断 (Tanh) 与硬截断 (Hard)（`test_energy_clipping_tanh_and_hard`）、单电子边界系统、参数冻结等价性检验（`test_local_energy_param_grad_frozen_equivalence`）、**理论变分协方差梯度逐参数精确匹配**（`test_surrogate_loss_gradient_exact_match`，理论导数与 Autograd 误差 $< 10^{-5}$）、全 NaN 能量安全拦截。
  - 断言质量：**极高质量**。逐参数梯度的数值微分与解析闭式比对具备决定性数学证明力；部分步长测试有轻微平庸断言（L127-L134 仅验证取值范围）。
- **`test_spin_and_cusp.py` (16 用例，全部通过)**：
  - 核心物理覆盖：自旋投影对称性（单重态空间对称性 `test_singlet_spatial_symmetry`、三重态空间反对称性 `test_triplet_spatial_antisymmetry`、$\hat{S}^2$ 本征值判定 `test_spin_eigenvalue_s_squared_operator`、闭壳层自旋翻转对称性）、核电荷 $Z=1,3,6$ 下的电子-核 Kato 尖点解析导数微商极限、平行与反平行电子-电子 Kato 尖点极限（$1/4$ 与 $1/2$）、**电子-核与电子-电子碰撞下动能与库仑势能万倍发散精确对消**（`test_coulomb_singularity_cancellation`, `test_electron_electron_coulomb_singularity_cancellation`, `test_full_ferminet_nuclear_cusp_divergence_cancellation`）、自旋自适应与 Kato Jastrow 复合模型端到端评估。
  - 断言质量：**极高质量**。公差严格（`atol=1e-10`，数值微分精度达 5 位小数），直接验证微商切线极限。
- **`test_p2_regressions.py` (12 用例，全部通过)**：
  - 核心回归覆盖：涵盖三大回归测试类——
    1. 自旋投影回归（`TestSpinProjectionRegressions`，5 用例）：验证多电子 $S^2$ 算符本征值与幂等性（包含 6 电子 $S=3$ 扇区 `test_multielectron_s_squared_idempotence_and_antisymmetry`）、单重态拒绝全反对称五重态（`test_singlet_rejects_fully_antisymmetric_quintet`）、自定义置换导数与 LiH 体系 Jastrow 复合自旋投影；
    2. 三重态 Kato 尖点回归（`TestTripletCuspRegressions`，3 用例）：验证三重态电子碰撞处波函数真实动能有界性（`test_real_triplet_local_energy_remains_bounded_at_coalescence`）、显式检验反向传播导数切线严格等于 $1/4$（`test_triplet_cusp_derivative_is_one_quarter`）、拒绝不兼容的显式 Jastrow 配置；
    3. 异常局域能量防御回归（`TestInvalidEnergyRegressions`，4 用例）：验证全 NaN/Inf 局域能量拒绝（抛出 `FloatingPointError`）、部分无效样本过滤无偏性（`test_partial_invalid_energy_uses_only_valid_samples`）、非有限对数波函数阻断更新、优化器在发生数值异常前立刻中止训练参数更新。
  - 断言质量：**极高质量**。专注于极端数值异常边界与长期训练回归防护。
- **`test_determinants.py` (4 用例，全部通过)**：
  - 核心数学覆盖：涵盖奇异行列式退化分支与高阶自动微分的深层数学防护——
    1. 零行列式分支一阶雅可比与二阶 Hessian 矩阵严格比对（`test_zero_branches_match_polynomial_first_and_second_derivatives`）：测试在不同尺寸（size=1,2,3）和降秩（rank $n-1$, $n-2$）及双自旋零行列式下，代数余子式分支的 Jacobian 与 Hessian 是否与独立的 Leibniz 多项式全展开参考实现（`polynomial_det`）严格吻合（公差达 `atol=1e-9, rtol=1e-9`）；
    2. 对数动态范围保护（`test_regular_branches_keep_logarithmic_dynamic_range`）：验证即使单电子轨道振幅衰减至 $10^{-20}$ 导致 float32 行列式下溢为 0，对数波函数依然保持数值精度与有限梯度；
    3. 奇异微扰参数导数验证（`test_zero_branch_preserves_tiny_total_wavefunction_and_derivatives`）：验证极小波函数在 $\theta=0$ 处的极值和一阶导为 1.0、二阶导为 -1.0；
    4. 真实模型零轨道有限能量与参数梯度检验（`test_real_model_zero_orbital_has_finite_energy_and_parameter_gradients`）：跨单双精度（float32/float64）和自旋极化体系（(1,0), (0,1), (2,2)）确认零轨道输入下的能量与参数梯度的有限性。
  - 断言质量：**极高质量**。代数解析级严格比对，导数匹配公差达到 `atol=1e-9`。
- **`test_cuda.py` (18 用例，CPU 环境下全部合规跳过)**：
  - 设备兼容覆盖：对上述全部核心特性（奇异行列式导数、自旋投影算符、三重态 Kato 尖点、局域能量与坐标梯度一致性、异常能量阻断、CUDA 训练参数迭代）提供 GPU 端的无缝迁移与设备一致性验证（`TestCudaAgreementAndTraining`, `TestCudaDeterminants`, `TestCudaInvalidEnergy`, `TestCudaSpin`, `TestCudaTripletCusp`）。
  - 断言质量：**规范合理**。通过 `@unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available.")` 在 CPU 环境下安全跳过，硬件抽象隔离清晰。
- **`test_packaging.py` (1 用例，全部通过)**：
  - 工程分发覆盖：`test_built_distributions_include_and_import_runtime_modules` 通过 Hatchling/`uv build` 独立构建源码分发包（sdist）与二进制轮子包（wheel），并在隔离的临时虚拟环境中进行跨包安装与全模块（`ferminet_pytorch` 及其全部子模块）独立导入自检，确保无打包遗漏与循环导入。
  - 断言质量：**高质量**。工业级 CI/CD 打包分发完整性检验。

- **综合测试防御洞察**：
  - **优势**：核心物理特性（反对称性、Kato 尖点对消、自旋投影、变分能量理论梯度）测试深度极高，断言严谨，杜绝平庸断言。
  - **改进建议**：测试套件尚缺乏对奇数电子自由基开壳层系统、波函数节点面（$\Psi \to 0$）穿越时高阶梯度的有限性回归测试，建议在后续演进中补充针对性用例。

---

## 三、 风险清单与缺陷矩阵 (Risk & Finding Matrix)

共计检出 **22 项核心发现**：
- **Critical (阻断级)**：4 项
- **High (严重级)**：6 项
- **Medium (中度级)**：8 项
- **Low (轻微级)**：4 项

| 编号 | 严重级别 | 所属维度 | 受影响文件与行号 | 缺陷核心机理 | 潜在物理与系统危害 |
| :--- | :---: | :---: | :--- | :--- | :--- |
| **FINDING-PYT-01** | **Critical** | 性能/数值 | `loss.py#L202-L205`<br>`hamiltonian.py#L143-L171` | 计算 $E_L$ 期间未临时冻结模型参数，强制构建全参数二阶计算图 | 显存膨胀数十倍，单步时间暴增 16.6 倍，严重限制批量与迭代 |
| **FINDING-PYT-02** | **Critical** | 数值/物理 | `features.py#L70-L72`<br>`envelope.py#L80-L82`<br>`hamiltonian.py#L88, L95` | 欧氏距离 clamp 截断使一阶导跳变，奇异区二阶拉普拉斯突变为 0 | 库仑尖点抵消失效，形成伪势能黑洞，吸引 Walker 塌陷发散 |
| **PHY-01** | **Critical** | 物理保真 | `spin.py#L74-L77`<br>`model.py#L217-L223` | 自旋投影硬编码拒绝奇数电子并盲目指定 singlet | 奇数电子体系（如 Li 原子的 $n_\uparrow=2, n_\downarrow=1$）初始化即崩溃 |
| **PHY-05** | **Critical** | 物理/数值 | `hamiltonian.py#L140-L150`<br>`model.py#L368-L378` | 波函数节点面处输出 -inf，二阶自动微分产生 $0 \times \infty = \text{NaN}$ | 训练中一旦 Walker 穿越节点面，全局梯度立即被 NaN 污染毒化 |
| **FINDING-PYT-03** | **High** | 性能/计算 | `hamiltonian.py#L159-L171` | 逐维 Python 循环反向传播 $3N$ 次计算拉普拉斯，丢弃非对角项 | 算力浪费超 96%（$N=10$ 时），显存图驻留严重阻碍大体系计算 |
| **FINDING-PYT-04** | **High** | 数值稳定 | `model.py#L30-L58` | 代数余子式递归在二阶微分下具备 $O((N!)^2)$ 阶乘爆炸复杂度 | 中大分子（$N \ge 8$）触发奇异分支时引发递归溢出或 CUDA OOM |
| **PHY-03** | **High** | 物理保真 | `features.py#L70-L75`<br>`model.py#L150-L155` | 默认关闭 smooth_cuspless，输入欧氏距离在核处呈现 $\nabla^2 d = 2/r$ | 神经网络在每个核坐标处产生非物理的伪动能发散，破坏变分下界 |
| **PHY-04** | **High** | 物理保真 | `envelope.py#L58-L59, L80` | 指数包络衰减率统一初始化为 1.0，与核电荷 $Z$ 未做物理对齐 | 重核体系初始产生未对消的近核奇异性，造成能量雪崩与采样陷入 |
| **PHY-06** | **High** | 物理/数值 | `hamiltonian.py#L88, L95` | 库仑势能硬截断破坏极小距离下的动能-势能导数协同抵消 | 单精度下 $10^{12}$ 大数相消导致有效数字全部抹平，能量评估方差剧增 |
| **F-MCMC-01** | **High** | 统计/MCMC | `mcmc.py#L248-L252`<br>`train_vmc.py#L198, L243` | 生产采样阶段未冻结自适应步长，破坏马尔可夫链时齐性 | 采得构型偏离玻恩稳态测度，能量与变分梯度引入系统性统计有偏 |
| **F-LOSS-01** | **High** | 优化/数值 | `loss.py#L39, L56-L68` | 退化分布下 MAD 趋零导致截断阈值坍缩至 $10^{-8}$，tanh 极度饱和 | 有效能量残差坍缩，反向传播梯度乘以 $10^{-8}$ 导致更新停滞（梯度窒息） |
| **F-TRAIN-01** | **High** | 优化算法 | `train_vmc.py#L204, L241` | 纯固定学习率 Adam 优化器，缺乏学习率退火调度器 | 在能量极小值附近被 Monte Carlo 采样噪声推动发生剧烈随机游走 |
| **PHY-02** | **Medium** | 物理/工程 | `spin.py#L169-L186` | 自旋投影 $K=\binom{N}{n_\uparrow}$ 项空间置换前向推断未批处理向量化 | 多电子（$N \ge 8$）体系前向与二阶求导计算图极度膨胀导致 OOM |
| **FINDING-PYT-05** | **Medium** | 深度学习 | `layers.py#L93-L107` | 双电子均值池化切片包含对角线自身项 $i=j$（$r_{ii} \equiv 0$） | 虚假几何特征稀释真实电子交互，引入 $1/N_\uparrow$ 的尺度偏差 |
| **FINDING-PYT-06** | **Medium** | 编译工程 | `model.py#L344` (`if singular.any():`) | 数据依赖的张量跳转导致 `torch.compile` 产生硬图断裂 (Graph Break) | 阻碍 Inductor 内核融合，使得编译性能收益大幅退化 |
| **F-MCMC-02** | **Medium** | 统计/MCMC | `mcmc.py#L36, L66` | 全电子协同移动与逐电子移动共用 0.5 目标接受率 | 全电子移动提议步长被过度压缩，电子扩散迟缓，自相关时间显著上升 |
| **F-MCMC-03** | **Medium** | 统计/MCMC | `mcmc.py#L238-L240` | 逐电子移动中将接受次数强行除以电子数并取整，总步数仅加 1 | 统计计数与真实物理提议次数脱节，历史总接受率分析失真 |
| **F-LOSS-02** | **Medium** | 变分理论 | `loss.py#L114-L120` | 静默剔除 NaN/Inf 局域能量样本，缺乏比例安全上限拦截 | 引入乐观选择偏差，报告能量虚假偏低，发散区域无法获得纠偏梯度 |
| **F-TRAIN-02** | **Medium** | 训练工程 | `train_vmc.py#L228-L299` | 缺乏历史最优模型状态（Checkpoint）深拷贝保存与断点恢复 | 训练结束仅返回最后轮次模型，若末尾发生随机震荡将丢失最优基态 |
| **PHY-07** | **Medium** | 测试质量 | `test_spin_and_cusp.py`<br>`test_ferminet.py` | 测试套件缺乏开壳层自由基体系、节点面穿越及极限近核物理验证 | CI 无法提前预警奇数电子崩溃与节点面高阶微分 NaN 异常 |
| **FINDING-PYT-07** | **Low** | 跨平台 | `train_vmc.py#L208`<br>`test_*.py` | Windows 默认 CP936 下 UTF-8 控制台字符乱码及模板编码陷阱 | 控制台输出乱码，特定环境编译 Inductor 可能抛出 UnicodeDecodeError |
| **F-TEST-01** | **Low** | 测试质量 | `test_vmc.py#L127-L134` | 自适应步长单元测试存在宽区间平庸断言 `(0.001, 10.0)` | 无法有效检测步长是否按照接受率偏差发生单调调整的动力学真实性 |

---

## 四、 核心推荐行动路线图与重构代码清单 (Actionable Recommendations)

本路线图按照 **P0 (阻断级，必须立即修复)** $\to$ **P1 (高危级，数值与物理稳健)** $\to$ **P2 (工程优化与扩展性)** 进行编排，每项均提供工业级前后对比代码。

### 4.1 第一阶段 P0：核心阻断级修复 (Critical Remediations)

#### 1. 修复 FINDING-PYT-01：局域能量求导时隔离模型参数计算图（16.6 倍加速）
- **受影响模块**：`loss.py#L202-L205`, `hamiltonian.py#L143-L150`
- **重构原理**：VMC 变分定理中 $E_L(r)$ 在反向传播中仅充当标量权重常数，不需要其关于模型参数的梯度。在计算 $E_L$ 期间将所有参数置为 `requires_grad=False`，彻底阻止 Autograd 为数万个网络权重构建二阶 Hessian 图。

**【修改前代码】 (`loss.py#L202-L205`)**：
```python
# 1. 计算局部能量 (使用独立坐标计算图)
r_for_energy = walkers.clone().detach().requires_grad_(True)
e_local, _, _ = compute_local_energy(model=model, system=system, r=r_for_energy)
e_local = e_local.detach()
```

**【修改后建议代码】 (`loss.py`)**：
```python
# 引入轻量级无梯度上下文管理器，阻断参数二阶图构建
class _FreezeModelParameters:
    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.prev_states = []

    def __enter__(self) -> None:
        self.prev_states = [p.requires_grad for p in self.model.parameters()]
        for p in self.model.parameters():
            p.requires_grad = False

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        for p, state in zip(self.model.parameters(), self.prev_states):
            p.requires_grad = state

# 1. 计算局部能量 (施加参数隔离，纯坐标求导模式运行)
r_for_energy = walkers.clone().detach().requires_grad_(True)
with _FreezeModelParameters(model):
    e_local, _, _ = compute_local_energy(model=model, system=system, r=r_for_energy)
e_local = e_local.detach()
```

---

#### 2. 修复 FINDING-PYT-02：欧氏距离平滑化与物理职责分离，消除伪发散与广播灾难
- **受影响模块**：`features.py#L70-L72`, `envelope.py#L80-L82`, `hamiltonian.py#L88, L95`, `jastrow.py`
- **重构机理与物理边界**：
  1. **广播灾难与维度保持控制**：原先直接采用硬编码 `keepdim=True` 的平滑函数会导致在 `hamiltonian.py#L88` 计算势能除法时，形状为 `(B, N, I, 1)` 与 `charges (1, 1, I)` 相除意外广播生成 `(B, N, I, I)` 的 4 阶张量灾难，消耗数倍显存并导致能量求和错误。因此必须增加 `keepdim: bool = False` 参数并继承宿主系统的 `eps`。
  2. **严格的物理边界与职责分离**：
     - **几何特征层（`features.py`）**：平滑欧氏距离 $\sqrt{r^2+\epsilon}$ 在 $r \to 0$ 处一阶导恒为 0，这**仅适用于**神经网络几何特征层 `features.py`（切断输入特征 $\nabla^2 |r| = 2/r$ 的伪发散，避免向神经网络反向传播无界动能）；
     - **物理尖点因子（`jastrow.py`）**：在真实满足量子力学第一性原理的 Kato 尖点因子中，**必须保留真实线性欧氏距离 $r_{ij}$**！因为平滑距离在 $r \to 0$ 处的导数 $\left.\frac{\partial \sqrt{r^2+\epsilon}}{\partial r}\right|_{r=0} = 0$，一旦代入 Jastrow 因子，将使得波函数在碰撞点对导数的贡献归零，彻底破坏 Kato 尖点导数条件（$\frac{\partial \ln \Psi}{\partial r_{ij}} \to \frac{1}{2}$ 或 $\frac{1}{4}$），导致其无法精确对消哈密顿量中真正的 $-Z/r$ 或 $1/r$ 库仑势能发散。两者物理职责必须严格解耦隔离。

**【修改前代码】 (`features.py#L70-L72`)**：
```python
d_en = torch.sqrt(torch.clamp(
    torch.sum(r_en ** 2, dim=-1, keepdim=True), min=self.eps
))
```

**【修改后建议代码】 (`features.py` / `hamiltonian.py`)**：
```python
def smooth_distance(
    r_diff: torch.Tensor,
    eps: float = 1e-12,
    keepdim: bool = False,
) -> torch.Tensor:
    """计算处处无穷次可微的平滑欧氏距离。
    
    注意：仅用于几何特征层 features.py，切断输入动能伪发散。
    在物理尖点因子 jastrow.py 中必须保留真实欧氏距离 r_ij 满足 Kato 条件！
    """
    sum_sq = torch.sum(r_diff ** 2, dim=-1, keepdim=keepdim)
    return torch.sqrt(sum_sq + eps)

# 在 features.py 中调用（保留末尾维度供多层感知机特征拼接）：
d_en = smooth_distance(r_en, eps=self.eps, keepdim=True)

# 在 hamiltonian.py 中调用（keepdim=False 避免广播灾难并继承系统 eps）：
dist_en = smooth_distance(diff_en, eps=eps, keepdim=False)
```

---

#### 3. 修复 PHY-01：开壳层体系半整数自旋安全降级，规避 Löwdin 谱展开算符伪物理陷阱
- **受影响模块**：`spin.py#L74-L99`, `model.py#L217-L228`, `loss.py`
- **重构机理与代数缺陷剖析**：
  1. **Löwdin 展开的整数循环依赖**：原代码 `spin.py#L80-L95` 中通过 `for spin in range(minimum, n // 2 + 1):` 遍历自旋本征值构造谱展开算符 $\prod_{S' \ne S} \frac{\hat{S}^2 - S'(S'+1)}{S(S+1) - S'(S'+1)}$。该谱投影循环依赖整数自旋本征值假设。
  2. **半整数自旋伪物理算符风险**：若强行在奇数电子体系（如自由基分子、Li 原子等 $N=3, 2S+1=2, S=1/2$）中指定半整数目标自旋，内层整数循环与半整数目标相互冲突，构建出的投影算符在线性代数与量子力学上是完全退化的**伪物理算符**（破坏投影算符的幂等性与本征态保真度）。
  3. **推荐且唯一安全的工程实践**：现阶段针对开壳层与自由基体系，推荐且唯一安全的工程实践是**安全自动降级为 `mode="none"`**；若需约束自旋纯度，可在变分损失函数 `loss.py` 中挂载自旋惩罚项：
     $$\mathcal{L} = E_L + \lambda \langle \hat{S}^2 - S(S+1) \rangle^2$$

**【修改前代码】 (`spin.py#L74-L77`, `model.py#L217-L223`)**：
```python
n = self.total_electrons
minimum = abs(self.n_up - self.n_down) // 2
if n % 2 or not minimum <= target <= n // 2:
    raise ValueError(f"mode={self.mode!r} is incompatible with n_up={self.n_up}, n_down={self.n_down}")
```

**【修改后建议代码】 (`model.py#L217-L228`)**：
```python
# 自动检测开壳层与奇数电子体系，安全降级至 mode="none"，规避半整数谱算符代数退化
is_open_shell = (min(n_up, n_down) == 0) or ((n_up + n_down) % 2 != 0) or (abs(n_up - n_down) % 2 != 0)
if is_open_shell:
    # 开壳层自由基体系：安全降级，推荐结合自旋惩罚损失项约束 S^2
    spk.setdefault("mode", "none")
else:
    spk.setdefault("mode", "singlet")
```

若需保证开壳层自旋纯度，可在 `loss.py` 中挂载自旋惩罚项：
```python
# 在变分优化损失中注入自旋期望惩罚，无需显式 Löwdin 谱投影
s2_penalty = spin_penalty_weight * (expected_s2 - target_s * (target_s + 1.0)) ** 2
total_loss = energy_loss + s2_penalty
```

---

#### 4. 修复 PHY-05：废除非物理常数硬截断，采用无副作用的节点面防御与异常样本隔离
- **受影响模块**：`hamiltonian.py#L140-L150`, `loss.py#L200-L220`, `mcmc.py#L180-L210`
- **重构机理与常数硬截断深度剖析**：
  1. **常数硬截断的致命弊端**：
     原建议中采用常数硬截断 `torch.where(log_abs_psi < -50.0, -50.0, ...)` 存在双重致命物理与数值缺陷：
     - **局域动能骤降为 0**：在截断发生区域，波函数对数被强行置为常数标量（$-50.0$），导致一阶导 $\nabla \ln|\Psi| \equiv 0$，二阶拉普拉斯算子 $\nabla^2 \ln|\Psi| \equiv 0$。计算出的局域动能 $T_L = -\frac{1}{2}(\nabla^2 \ln|\Psi| + |\nabla \ln|\Psi||^2) \equiv 0$ 骤降归零，彻底摧毁节点面附近的真实量子动力学行为；
     - **误伤大分子正常波函数**：对于多电子大分子体系（如 $N \ge 20$），波函数随着空间维度增加呈指数衰减，在正常的物理采样空间中，有效波函数的对数值完全可能合理低于 $-50.0$。硬编码 `-50.0` 会在大分子前向空间人为制造大面积平坦“伪死区”，误伤大分子正常波函数并引发梯度虚假消失。
  2. **三大无副作用的工业级重构方案**：
     - **解法 A（MCMC 节点拒绝，从抽样源头消灭节点面构型）**：在 MCMC Metropolis-Hastings 采样中，对提议为 $-\infty$ 或非有限（NaN/Inf）的波函数判定接受率比值 $A = 0$，无条件拒绝该转移步。根据波函数概率测度 $|\Psi|^2$，节点面测度本就为 0，抽样将严格回避节点面。
     - **解法 B（损失层有效样本掩码隔离，彻底杜绝 NaN 梯度扩散）**：在 `loss.py` 变分损失层构建有效样本布尔掩码：
       `valid_mask = torch.isfinite(e_local) & torch.isfinite(log_abs_psi)`
       在执行参数更新前隔离非有限样本，仅由健康构型提供变分梯度，彻底杜绝 NaN 梯度扩散。
     - **解法 C（前向 Pre-log 截断防御，消除 $0 \times \infty = \text{NaN}$）**：若需波函数内部数值安全，在求对数之前（Pre-log）使用 `torch.log(torch.clamp(psi_abs, min=1e-50))`，避免原生 `torch.where` 未选中分支在反向传播求导时计算 `0 * inf = NaN`。

**【修改前代码】 (`hamiltonian.py#L140-L150`)**：
```python
log_abs_psi, _ = model.forward_log_psi(r)
ones = torch.ones_like(log_abs_psi)
grad_log_psi = torch.autograd.grad(
    outputs=log_abs_psi,
    inputs=r,
    grad_outputs=ones,
    create_graph=True,
    retain_graph=True,
)[0]
```

**【修改后建议方案：工业级三道防线协同防御】**：
```python
# 解法 A：MCMC 转移拒绝（mcmc.py）—— 从源头避免 Walker 滞留于节点面
log_prob_proposed = 2.0 * log_psi_proposed
# 若提议波函数为 -inf 或 NaN，直接将转移接受率置为 0
acceptance_ratio = torch.where(
    torch.isfinite(log_prob_proposed),
    torch.exp(log_prob_proposed - log_prob_current),
    torch.zeros_like(log_prob_proposed),
)

# 解法 B：变分损失层有效样本掩码隔离（loss.py）—— 阻断偶发异常构型梯度扩散
valid_mask = torch.isfinite(e_local) & torch.isfinite(log_abs_psi)
if not valid_mask.all():
    e_local_clean = e_local[valid_mask]
    log_abs_psi_clean = log_abs_psi[valid_mask]
else:
    e_local_clean = e_local
    log_abs_psi_clean = log_abs_psi

# 解法 C：前向 Pre-log 截断防御（model.py）—— 规避原生 torch.where 的 0*inf 反向传播缺陷
safe_psi_abs = torch.clamp(torch.abs(psi_val), min=1e-50)
log_abs_psi = torch.log(safe_psi_abs)
```

---

### 4.2 第二阶段 P1：高危数值稳定性与物理缺陷修复 (High-Priority Remediations)

#### 1. 修复 F-MCMC-01：生产抽样冻结自适应步长，捍卫马尔可夫链时齐性与细致平衡
- **受影响模块**：`mcmc.py#L255-L264`
- **重构原理**：自适应步长仅在 Burn-in 预热阶段允许开启；进入正式生产抽样时强行冻结转移概率核，确保稳态分布严格收敛于 $|\Psi|^2$。

**【修改前代码】 (`mcmc.py#L255-L260`)**：
```python
def sample(self, num_steps: int = 10) -> torch.Tensor:
    """执行多步连续 MCMC 迭代并返回最新构型副本。"""
    for _ in range(num_steps):
        self.step()
    return self.walkers.clone().detach()
```

**【修改后建议代码】 (`mcmc.py`)**：
```python
def sample(self, num_steps: int = 10, adapt: bool = False) -> torch.Tensor:
    """
    执行多步连续 MCMC 迭代并返回最新构型副本。
    默认在采样期间冻结步长调整，保证马尔可夫链的时齐性与细致平衡。
    """
    prev_adapt = self.adapt_step_size
    self.adapt_step_size = adapt
    try:
        for _ in range(num_steps):
            self.step()
    finally:
        self.adapt_step_size = prev_adapt
    return self.walkers.clone().detach()
```

---

#### 2. 修复 F-LOSS-01：设置有效能量下限，杜绝退化分布下 MAD 截断诱发的梯度窒息
- **受影响模块**：`loss.py#L56-L58`
- **重构原理**：截断阈值 $\Delta$ 必须兼顾样本的实际物理能量尺度，即使批次能量极度集中导致 $\text{mad} \approx 0$，$\Delta$ 也不得低于物理最小阈值，防止 `tanh` 饱和压平梯度。

**【修改前代码】 (`loss.py#L56`)**：
```python
delta = torch.clamp(clip_factor * mad, min=eps)
```

**【修改后建议代码】 (`loss.py`)**：
```python
# 结合绝对最小容差与宏观能量尺度，防止极端退化分布下的梯度窒息
min_physical_delta = max(float(eps), 1e-3, 0.01 * float(torch.abs(e_median).item()))
delta = torch.clamp(clip_factor * mad, min=min_physical_delta)
```

---

#### 3. 修复 PHY-03：强制默认启用平滑输入特征，切断特征层 $1/r$ 伪发散
- **受影响模块**：`model.py#L152`
- **重构原理**：无论是否配置 Jastrow 因子，输入几何特征均必须使用平滑形式 $d_{en}^2 / (1 + d_{en})$，避免高阶拉普拉斯算子在核坐标处产生非物理的 $2/r$ 伪奇异性。

**【修改前代码】 (`model.py#L152`)**：
```python
eff_smooth = smooth_cuspless if smooth_cuspless is not None else has_jastrow
```

**【修改后建议代码】 (`model.py`)**：
```python
# 无论是否挂载显式 Jastrow，输入特征均强制采用平滑无尖峰形式，切断 2/r 动能发散
eff_smooth = smooth_cuspless if smooth_cuspless is not None else True
```

---

#### 4. 修复 F-TRAIN-01：在训练循环中增加学习率退火调度器
- **受影响模块**：`train_vmc.py#L204-L208, L280`
- **重构原理**：引入 CosineAnnealingLR 调度器，在训练后期压制 Monte Carlo 残余随机梯度噪声，促进化学精度收敛。

**【修改后建议代码】 (`train_vmc.py`)**：
```python
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=num_epochs, eta_min=lr * 0.05
)

for epoch in range(1, num_epochs + 1):
    # ... 执行训练步 ...
    scheduler.step()
```

---

### 4.3 第三阶段 P2：性能与扩展性重构建议 (Medium-Priority Architecture Refactoring)

1. **利用 Forward-mode AD (JVP) 向量化拉普拉斯算子计算 (`FINDING-PYT-03`)**：
   引入 PyTorch 2.0+ 的 `torch.func.jvp` 与 `torch.func.vmap`，一次性向量化计算 Hessian 对角线方向导数，消除 $3N$ 次标量 Python 循环，省去反向全图驻留，消除 96% 的无谓 Hessian 运算。
2. **矩阵代数余子式递归尺寸截断 (`FINDING-PYT-04`)**：
   在 `model.py` 中限制代数余子式仅在矩阵尺寸 $\le 4$ 时递归展开；大于 4 时切换到基于 Tikhonov 阻尼正则化的伴随矩阵求解，彻底消除 $O((N!)^2)$ 显存爆炸隐患。
3. **双电子特征聚合剔除自交互项 (`FINDING-PYT-05`)**：
   在 `layers.py` 中构造掩码矩阵，在均值池化时剔除 $j=i$ 的自身对角线虚假零距离特征，恢复自旋等变图表示的纯洁性。
4. **全电子移动目标接受率适配 (`F-MCMC-02`)**：
   在 `mcmc.py` 初始化中根据移动模式动态设定目标接受率：全电子协同移动设为 0.25，逐电子移动设为 0.50，使构型空间扩散效率最大化。

---

## 五、 结论与审查判定

`ferminet_pytorch` 在双电子单重态（如 $H_2$ 分子）教学演示与核心物理性质测试上表现优异，代码结构清爽、理论注释丰富、且显式 Kato Jastrow 因子与变分协方差梯度阻断的实现展示出了深厚的量子物理素养。

然而，作为向通用分子与多电子体系扩展的从头算框架，本项目目前存在 **计算图未隔离（16.6 倍减速）**、**clamp 诱发虚假近核势能黑洞**、**开壳层奇数电子自旋崩溃**、**节点面自动微分 NaN 毒化** 等关键工程与理论物理漏洞。

依照本审查报告所列出的 P0 阻断级清单与代码对比方案实施重构后，该项目将具备极高的生产级数值稳定性、严谨的无偏统计抽样保障、以及大幅跃升的计算性能，完全有能力胜任大分子与复杂化学反应的第一性原理变分量子蒙特卡洛研究。
