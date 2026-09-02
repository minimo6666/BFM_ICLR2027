# BFM V9：最小 terminal boundary-repair 训练

## 1. 这次只验证一个假设

Oracle waterfall 已把 V8 的主要 realization gap 定位到：

- 64 NFE：`2 -> 1`；
- 32 NFE：`3 -> 1`；
- `1 -> 0` hard final 会继续放大误差，但不是第一根因。

V8 在最后 probabilistic transition 使用 sensitivity adapter，却在物理
`t=1` 强制退回 base predictor。因此本实验只增加一个 terminal
posterior-carry head：

```text
adapter-corrected m_t + sampled X_1 + source interval
                         -> corrected terminal clean logits -> same hard final
```

V8 backbone 和原 sensitivity adapter 全部冻结；BFM 解析 reverse posterior、
采样 grid、temperature 和 hard final 全部不改。

第一轮 head 只对 64/32 NFE 启用；16/8 NFE 自动保持原 V8 路径，禁止用未训练
的 source timestep 做外推。

## 2. 为什么训练分成两类 branch

Dynamic consistency 使用模型自己的边缘化 reverse kernel：

$$
X_1^{(b)}\sim P_\theta(X_1\mid X_t),\qquad
L_{\rm prob}=\left\|\frac1B\sum_b\tilde m_1(X_1^{(b)})-
\operatorname{sg}(\hat m_t)\right\|_2^2.
$$

Hard-aware consistency 对齐真正部署的 hard endpoint：

$$
L_{\rm hard}=\left\|\frac1B\sum_b
\sigma\!\left(\tilde\ell_1(X_1^{(b)})/\tau\right)-
\operatorname{sg}(\hat m_t)\right\|_2^2.
$$

Terminal BCE 不能把上述边缘化样本随便重新配回数据中的同一张 $X_0$。
因此 BCE 使用给定 paired clean bit 的精确 bridge：

$$
\bar X_1^{(b)}\sim q(X_1\mid X_t,X_0),\qquad
L_{\rm BCE}=\frac1B\sum_b
\operatorname{BCE}(\tilde m_1(\bar X_1^{(b)}),X_0).
$$

总损失固定为：

$$
L=L_{\rm BCE}+L_{\rm prob}+L_{\rm hard}.
$$

第一轮不做 lambda sweep。

## 3. 文件

```text
models/
  transformer_boundary_repair_v9.py
  binarylatent_flow_boundary_repair_v9.py

experiments/ICLR27/new_theory_sensity_aware_loss/bfm_v9_boundary_repair/
  train_boundary_repair_v9.py
  evaluate_boundary_repair_v9.py
  generate_v9_10k.py
  summarize_fid10k.py
  v9_runtime.py
  run_train_gpu01.sh
  run_eval_gpu0.sh
  run_fid10k_paired_gpu01.sh
  README_CN.md
```

上述目录已经包含 V9 验证和采样所需的运行时辅助函数，不依赖未上传的
V8 post-hoc analysis 目录。FID shell 脚本复用仓库现有的 Churches FID
评测入口；若仅训练 V9，则不需要该评测入口。

## 4. GPU 0/1 小量训练

必须使用与 V8 post-hoc 完全相同的 EMA 50k checkpoint：

```bash
cd /mnt/data/b/mohao/Projects/BinaryLatentDiffusion

export V8_CHECKPOINT=/绝对路径/flow_lowrank_sensitivity_v8_ema_50000.th
export AE_LOAD_DIR=logs/BAE_C64
export DATA_ROOT=/mnt/data/0/mohao/data/lsun/scenes

bash experiments/ICLR27/new_theory_sensity_aware_loss/bfm_v9_boundary_repair/run_train_gpu01.sh
```

默认配置：

| 项目 | 值 |
|---|---:|
| GPU | physical 0,1 |
| updates | 5,000 |
| per-GPU batch | 8 |
| dynamic branches | 4，antithetic |
| trainable module | terminal carry head only |
| optimizer | AdamW, lr=1e-3 |
| EMA | head only, 0.995 |
| checkpoint | 每 500 updates |
| backbone / V8 adapter | frozen |
| hard final | unchanged |

如显存不够，只改：

```bash
export BATCH_SIZE=4
export BFM_V9_TERMINAL_CHUNK=8
```

不要先改 loss 权重。

训练输出：

```text
runs/01_boundary_repair_64_32/
  config.json
  metrics.jsonl
  checkpoints/
    boundary_head_step_500.pt
    ...
    boundary_head_latest.pt
```

checkpoint 只保存约几万参数的 boundary head 和 optimizer，不重复保存整个
171M V8 checkpoint。评估时必须同时提供原 V8 checkpoint。

## 5. 先做 boundary validation，不直接跑 FID

```bash
export V8_CKPT="$V8_CHECKPOINT"
export AE_CKPT=/绝对路径/binaryae_ema_8100000.th
export DATA_ROOT=/mnt/data/0/mohao/data/lsun/scenes

bash experiments/ICLR27/new_theory_sensity_aware_loss/bfm_v9_boundary_repair/run_eval_gpu0.sh
```

默认使用 128 张 validation images、每个 anchor 32 个 branch，并用相同随机数
严格配对比较：

- `v8_head_off`：同一 V8 checkpoint，关闭新 head；
- `v9_head_on`：加载 EMA boundary head；
- `delta_v9_minus_v8`：配对差值。

主要文件：

```text
runs/02_boundary_validation/boundary_metrics.csv
runs/02_boundary_validation/decision.json
```

## 6. 预先固定的继续/停止标准

每个 NFE 必须同时满足：

1. `hard_realization_gap` 至少降低 50%；
2. `endpoint_hard_brier` 低于 head-off V8；
3. `teacher_local_brier` 配对差异小于 `1e-6`，证明上游 posterior 未改变。

`decision.json` 中对应 `pass_for_10k_fid=true` 时，才值得跑该 NFE 的
10k FID。若 2k updates 已经明显满足，可提前评估 2k checkpoint；若 5k 后仍
不满足，不扩展训练或 sweep，说明仅靠 terminal carry head 的表达能力不足，
下一步才允许 transition kernel 一起接受 dynamic-consistency 梯度。

## 7. 当前结论边界

这次成功只能证明：修复最后 boundary 能降低多步 posterior realization gap，
并为恢复 64/32 FID 提供因果证据。它不能在跑 FID 前声称 FID 必然改善，也
不能仅凭逐 bit dynamic consistency 保证完整 joint distribution 正确。
