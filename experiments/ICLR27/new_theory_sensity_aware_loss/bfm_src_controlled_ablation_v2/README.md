# BFM SRC Controlled Ablation v2

这版刻意把实验缩窄：**先只验证 loss，不改 sampling algorithm。**

---

# 一句话目的

当前 corrected BLD ≈ BFM 已经成立。现在只问：

> 在正确的 BFM reverse sampler 不变的前提下，
> sensitivity/error-transmissivity 推导出的 loss 是否能训练出更好的 predictor，
> 从而提高标准 64-NFE sampling quality？

第一轮不做 adaptive sampling，不做 multi-grid loss，不做第二次 forward，
不改 final hard projection。

---

# 为什么上一版太复杂

上一版同时引入：
- t 对齐；
- multi-grid deployment interval；
- 第二个 SRC forward；
- 新 loss。

即使最后 FID 上升，也不容易说清是谁造成的。

v2 改成严格分阶段。

---

# Experiment 0：先单独解决 t 问题

## 0A Current/original BFM

训练：
`time_steps=t`

采样：
`time_steps=t`

loss：
原 base BCE/flip objective

posterior：
corrected expectation-consistent

运行：

```bash
bash experiments/ICLR27/src_controlled/train/run_00_original_bfm.sh
```

## 0B Aligned BFM

唯一概念变化：

`time_steps=t-1`

同时用于训练和采样。

loss 完全不加 SRC。

运行：

```bash
bash experiments/ICLR27/src_controlled/train/run_01_aligned_bce.sh
```

比较 0A vs 0B，是单独的 **time-convention ablation**。

之后所有新 loss 都以 0B 为 baseline。

---

# Experiment 1：真正的 loss ablation

从这里开始，三个模型的 sampling code 完全相同：

- corrected expectation-consistent posterior
- `time_steps=t-1`
- 同一个 np.linspace sampling grid
- 同一个 Bernoulli draw
- 同一个 hard final
- 不做任何 adaptive/sensitivity-aware sampling

只改训练 loss。

## A. Aligned BCE baseline

```text
L = L_base
```

就是 `run_01_aligned_bce.sh`。

## B. Plain-Brier control

```text
L = L_base + lambda * Brier
```

没有 sensitivity。

目的：
如果加一个额外的 MSE/Brier regularizer 本身就能提升，
我们不能把提升归因于 S。

运行：

```bash
bash experiments/ICLR27/src_controlled/train/run_02_aligned_plain_brier.sh
```

## C. Proposed SRC

```text
L = L_base
  + lambda * [S(t-1,t)^2 / E(S^2)] * Brier
```

这里：
- `s=t-1`
- 使用和 64-NFE sampler 完全匹配的 one-step interval
- t=1 -> s=0 因为当前是 hard final，所以 SRC 权重设为 0
- 不 clipping
- `1/E(S^2)` 只是全局正比例常数，不改变 sensitivity 的相对权重

运行：

```bash
bash experiments/ICLR27/src_controlled/train/run_03_aligned_src.sh
```

---

# 为什么这次不用第二个 forward？

Baseline、plain-Brier、SRC 全部：

1. 同样 uniform sample 一个 t；
2. 同样生成一个 X_t；
3. 同样只 forward 网络一次；
4. 得到同一个 m_theta；
5. 然后 loss 不同。

所以：
- compute budget 基本一致；
- 看过的 noisy states 数量一致；
- timestep sampling distribution 一致。

这样 plain-Brier vs SRC 的**唯一差异就是 sensitivity weighting**。

---

# 为什么第一轮只用 one-step S(t-1,t)？

因为它与当前 64-NFE sampler严格匹配：

```text
64->63->62->...->2->1->0
```

我们现在首先想验证 sensitivity loss 本身有没有真实效果。

如果一上来做 8/16/32/64 multi-grid，
又引入 arbitrary interval training，
解释会变复杂。

所以第一轮 primary metric 是 **64 NFE**。

4/8/16/32 NFE 可以顺手评估，但它们属于“未匹配 low-NFE generalization”，
不是这轮理论的 primary claim。

如果 64-NFE SRC 有正 signal，下一轮再单独做
low-NFE interval-matched SRC。

---

# 运行顺序

建议不要一次全开。

## 第一步：先看 t 是否有影响

```bash
TRAIN_STEPS=50000 \
bash experiments/ICLR27/src_controlled/train/run_00_original_bfm.sh

TRAIN_STEPS=50000 \
bash experiments/ICLR27/src_controlled/train/run_01_aligned_bce.sh
```

同样 64-NFE FID。

如果两者接近，说明 t convention 不是主要性能来源；
但以后统一使用 t-1 baseline。

## 第二步：只验证 extra Brier

```bash
TRAIN_STEPS=50000 \
BFM_SRC_LAMBDA=1.0 \
bash experiments/ICLR27/src_controlled/train/run_02_aligned_plain_brier.sh
```

## 第三步：验证 sensitivity weighting

```bash
TRAIN_STEPS=50000 \
BFM_SRC_LAMBDA=1.0 \
bash experiments/ICLR27/src_controlled/train/run_03_aligned_src.sh
```

---

# 最重要的结果表

第一轮先只填 64 NFE：

| Model | t convention | Training loss | Sampling | 64-NFE FID |
|---|---|---|---|---:|
| Original BFM | t | base | original corrected BFM / t | ? |
| Aligned BCE | t-1 | base | corrected BFM / t-1 | ? |
| Aligned BCE + Brier | t-1 | base + Brier | **same as aligned BCE** | ? |
| Aligned BCE + SRC | t-1 | base + S²-Brier | **same as aligned BCE** | ? |

解释：

- Original vs Aligned BCE：只看 t。
- Aligned BCE vs Plain-Brier：看“额外 Brier”本身。
- Plain-Brier vs SRC：看真正的 sensitivity weighting。
- Aligned BCE vs SRC：看完整 proposed loss gain。

---

# 我们真正希望看到什么？

最理想：

```text
Aligned BCE            6.8x
Plain Brier             6.8x  （接近）
SRC                     明显更低
```

这样说明：
不是随便多一个 loss 就有效，
而是 sensitivity weighting 有效。

如果：

```text
Plain Brier ≈ SRC < BCE
```

说明主要是 Brier regularization，不足以支持 sensitivity claim。

如果：

```text
BCE ≈ Brier ≈ SRC
```

说明 one-step sensitivity reweighting在真实模型中没有明显作用，
下一步再考虑 low-NFE interval-matched risk。

如果：

```text
SRC > BCE
```

说明当前 weighting 可能过强或理论风险并不是当前视觉瓶颈，
先分析 high-S prediction error 再调 lambda，不能直接继续堆复杂 sampler。

---

# Sampling 这轮完全不改

这是最关键的一条。

这轮**没有“新采样方法”**。

Aligned BCE、Plain-Brier、SRC 都继承完全同一个：

```text
corrected expectation-consistent BFM sampler
time_steps=t-1
same grid
same hard final
```

所以如果 Plain-Brier 与 SRC FID 不同，
不能归因于 sampling algorithm。

---

# 与 loss 匹配的 sampling 是什么意思？

SRC 这轮用的是：

```text
S(t-1,t)
```

因此和它严格匹配的是标准 **64-NFE one-step sampler**。

所以 primary test：

```text
64 NFE
```

就是“loss-matched sampling”。

而 32/16/8/4 NFE 是额外 generalization test，
因为它们使用更大的 interval，并没有在这一版 loss 中显式匹配。

---

# Final hard projection

仍然保持当前：

```text
m_theta > 0.5
```

SRC 不对 `1->0` 加权。

所以这一轮也没有把 final stochastic sampling 混进来。

如果后面发现 artifact 几乎都来自 final hard projection，
再单独做 Experiment 2，不和 loss 实验混在一起。

---

# 先跑 sanity

```bash
python experiments/ICLR27/src_controlled/sanity/inspect_one_step_src.py
```

你会看到不同 t 的 S² 权重。

---

# 文件

```text
models/
  binarylatent_flow_controlled_src.py

experiments/ICLR27/src_controlled/
  train/
    train_controlled_src.py
    common.sh
    run_00_original_bfm.sh
    run_01_aligned_bce.sh
    run_02_aligned_plain_brier.sh
    run_03_aligned_src.sh

  sanity/
    inspect_one_step_src.py
```

---

# 给 Codex 的核心要求

不要修改 sampler 逻辑来“配合”SRC。

第一轮唯一允许的差异：

1. Experiment 0：t vs t-1；
2. Experiment 1：base / plain-Brier / S²-Brier。

尤其禁止同时：
- 改 sampling schedule；
- 改 hard final；
- 加 old aux posterior CE；
- 改 data sampling；
- 增加第二个 forward；
- 用不同 NFE 作为训练路径。

否则无法归因。
