# FermiNet 裁剪 × 包络受控实验

固定源码：`9b001c5bbd4c143058248bbc0c6e3585412ae27c`；从三个 50000 轮父检查点分叉，每组 10000 次更新。
本结果仅比较已训练 Gaussian 解的迁移，不代表从头训练或其他主干。

| 组别 | 种子 | 状态 | 参考偏差 mHa | SEM mHa | 化学精度 | 采样诊断 |
|---|---:|---|---:|---:|---|---|
| A | 11 | completed | 0.742464 | 0.052624 | met | passed |
| A | 22 | completed | 1.598052 | 0.058322 | not_met | passed |
| A | 33 | completed | 1.103957 | 0.047988 | met | passed |
| B | 11 | completed | 0.709708 | 0.043338 | met | passed |
| B | 22 | completed | 1.277287 | 0.045738 | met | passed |
| B | 33 | completed | 0.863300 | 0.057100 | met | passed |
| C | 11 | completed | 0.022521 | 0.030761 | met | passed |
| C | 22 | completed | 0.004478 | 0.031614 | met | passed |
| C | 33 | completed | 0.032653 | 0.039777 | met | passed |
| D | 11 | completed | 0.031428 | 0.026323 | met | passed |
| D | 22 | completed | 0.015295 | 0.026488 | met | passed |
| D | 33 | completed | 0.026221 | 0.026478 | met | passed |

交叉实验统计状态：**complete**。

| 效应 | 配对均值 mHa | Bonferroni 近似区间 mHa | 达到预设改善标准 |
|---|---:|---|---|
| clipping | -0.116031 | [-0.394219, 0.162157] | False |
| envelope | -1.046245 | [-2.606224, 0.513734] | False |
| interaction | 0.164057 | [-0.617823, 0.945938] | False |

四组为 A Gaussian+tanh、B Gaussian+mean_hard、C soft_exponential+tanh、D soft_exponential+mean_hard。
负的主效应表示能量降低；交互项表示非加性，不能独立解释为某个方案的总收益。
三种子的配对区间为近似区间，采用种子间标准误与传播 Monte Carlo 标准误的较大者。
裁剪对照改变一整套规则，不能拆分为尺度或硬裁剪的单独因果结论。

原始零步评估、最终空间尾部贡献、数值审计和所有状态见 verified_summary.json 及 artifacts/。
缺组、失败或采样诊断不足时不发布完整交叉实验结论。大型检查点与坐标分片保留在 source_download 指定目录。
