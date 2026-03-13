# ASD 与混合精度共识（与实现对照）

本文档整理对话共识，并对照 ASDQ 代码说明已实现部分与使用方式。

---

## 0. 分组共识

- **一个 group 是什么**：**一行 × 一坨列**。  
  权重矩阵形状 `(out_features, in_features)`，例如 `w_group=128` 时，把矩阵按**行**切成很多段，每段是**同一行上的 128 个连续列**，即形状上是一块 **(1, 128)**。
- **量化参数**：每个 group 有**自己**的 scale / zero_point；不同 group 的量化参数不同。

**代码对应**：`pseudo_quantize_tensor` 里 `tensor.reshape(-1, q_group_size)` 把权重拉成 `(out_f * in_f / 128, 128)`，每一行就是一个 group。

---

## 1. ASD 公式

```
ASD_c = θ1 × K_c + θ2 × Ψ_c
```

对**每个输入通道 c** 算一个 ASD 值。ASD 越高的通道，在量化时保留对应**权重列**的原始精度。

- **θ1 = 0.8**（默认），K 的权重
- **θ2 = 0.2**（默认），Ψ 的权重
- θ1、θ2 可在 `configs/default.yaml` 中配置

---

## 2. K：绝对显著性（Hessian-based importance）

### 定义

```
importance_c = ||W[:, c]||² × diag(H)_c
```

其中：
- `||W[:, c]||²`：权重第 c 列的平方和（"层对该通道的依赖程度"）
- `diag(H)_c = E[x_c²]`：Hessian 对角线（"该通道激活的能量"）

### 物理含义

度量"量化第 c 列权重对输出方差造成多大损失"。推导来自 Optimal Brain Surgeon 理论：

```
ΔL_c ∝ ||δW[:, c]||² × E[x_c²] ∝ ||W[:, c]||² × E[x_c²]
```

### 跨层可比性

importance 度量的是"对输出方差的贡献"，不同层的值天然同单位、可直接比较。

### 归一化

K_normalized = importance / global_max，全模型所有通道的 importance 一起归一化到 [0, 1]。

**代码**：`asdq/metrics/asd.py` → `compute_importance(weight, diag_H)`

---

## 3. Ψ：相对显著性（层内 z-score）

### 定义

```
Ψ_c = max(0, (importance_c − μ_layer) / σ_layer)
```

### 含义

"该通道在同层内有多突出"。z-score 衡量偏离层均值几个标准差；负值 clamp 为 0（低于均值 = 不突出）。

### 归一化

Ψ_normalized = Ψ / global_max_Ψ，所有层的 Ψ 一起归一化到 [0, 1]，与 K_normalized 可加权合并。

**代码**：`asdq/metrics/asd.py` → `compute_Psi(importance, method="zscore")`

---

## 4. 流式 Hessian 对角收集

### 问题

旧方案存储原始激活 [N, C]，显存 O(层数 × N × C)，大模型会爆。

### 方案

流式统计 diag(H)_c = E[x_c²]，只需累加器：

```python
sum_x2[key] += x.pow(2).sum(dim=0)   # shape [C]
count[key] += N_tokens
# 最终: diag_H = sum_x2 / count
```

显存降为 O(层数 × C)。

### 与 OWQ/GPTQ 的关系

OWQ 的 `add_batch` 计算完整 Hessian 矩阵 H (C×C)，我们只需对角线 diag(H)。数学等价：`diag(X @ X.T)_c = Σ_n x_{n,c}²`。我们跳过非对角线计算，内存从 O(C²) 降到 O(C)。

**代码**：`asdq/calibration/hessian_collector.py` → `collect_hessian_diag(...)`

---

## 5. 混合精度（SpQR 风格）

### 原则

ASD 越高 → 该通道对应权重列保留原始 float 精度（outlier）；其余列量化为 `asd_low_w_bit`（默认 4-bit）。

### 全局比例上限

设全局 outlier 比例 `asd_high_precision_ratio`（默认 0.1）。所有层所有通道一起排序，取前 10%，不是每层各自 10%。

### SpQR 分组量化

分组 = 一行 × w_group 列（如 128 列）。组内落在高精度列上的权重视为 outlier：

1. 组内把 outlier 位置用该行非 outlier 位置的均值填充
2. 用填充后的 group 算 scale / zero_point
3. 对整组做量化→反量化
4. outlier 位置写回原始 float 权重

输出 = 量化结果 × (1 − mask) + 原权重 × mask。推理时直接 matmul。

### 特殊情况

某行某 group 全是保存列：整段不量化，当保存精度（极罕见）。

**代码**：
- `asdq/quantization/quant_funcs.py` → `pseudo_quantize_weight_spqr_style`
- `asdq/quantization/quantize.py` → `pseudo_quantize_model_weight`

---

## 6. 全流程

| 阶段 | 操作 | 代码 |
|------|------|------|
| 准备 | 读配置 + 加载模型 | `main_quant.py` |
| 校准数据 | 加载 COCO 格式数据 | `asdq/calibration/coco_vl.py` |
| 收集 Hessian 对角 | 前向钩子流式累加 E[x_c²] | `asdq/calibration/hessian_collector.py` |
| 计算 ASD | importance → K(全局归一化) + Psi(层内z-score) → ASD | `asdq/metrics/asd.py` + `asdq/quantization/mixed_precision.py` |
| 选择高精度列 | 全局排序取 top ratio | `mixed_precision.select_high_precision_columns` |
| SpQR 量化 | 分组内剔除 outlier → scale/zero → 量化 → 合并原值 | `asdq/quantization/quantize.py` |
| 保存 | 合并后 float 权重存 state_dict | `main_quant.py` |
| 评估 | 加载 scale_path 跑 lmms-eval | `main_eval.py` |

---

## 7. 配置项

```yaml
# ASD 参数
asd_theta1: 0.8              # K（绝对显著性）的权重
asd_theta2: 0.2              # Psi（相对显著性）的权重
asd_normalize: true           # 归一化 K/Psi 到 [0,1]

# 混合精度
asd_mixed_precision: true     # 是否启用（默认启用）
asd_high_precision_ratio: 0.1 # 全局保留精度比例
asd_low_w_bit: 4              # 非 outlier 列的量化比特

# 量化
w_bit: 4                      # 统一量化比特（mixed_precision=false 时用）
w_group: 128                  # SpQR 分组大小
```

层 key 约定：校准与量化统一使用 `layers.{block_idx}.{linear_name}`。

---

## 8. 归一化总结

| 归一化 | 对谁做 | 次数 | 作用 |
|--------|--------|------|------|
| K 全局归一化 | importance / global_max | 1 次全局 | 不同层的绝对重要性可比，落入 [0,1] |
| Psi z-score | (importance − 层均值) / 层标准差 | 每层 1 次 | 衡量层内异常程度 |
| Psi 全局归一化 | Psi_raw / global_max_Psi | 1 次全局 | Psi 落入 [0,1]，与 K 可加权合并 |
