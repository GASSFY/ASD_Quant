# ASD 与混合精度共识（与实现对照）

本文档整理对话共识，并对照 ASDQ 代码说明已实现部分与使用方式。

---

## 0. 分组共识（先对齐）

- **一个 group 是什么**：**一行 × 一坨列**。  
  权重矩阵形状 `(out_features, in_features)`，例如 `q_group_size=128` 时，把矩阵按**行**切成很多段，每段是**同一行上的 128 个连续列**，即形状上是一块 **(1, 128)**。  
  所以：**分组不是按列**，而是「一行上的一坨列」为一个 group。
- **量化参数**：每个 group 有**自己**的 scale / zero_point；不同 group 的量化参数不同。  
  因此沿**同一行**会有多段（多个 group），**每一段的量化参数不一样**（一行有多组，每组一套参数）。

**代码对应（非混合路径）**：`pseudo_quantize_tensor` 里 `tensor.reshape(-1, q_group_size)` 把权重拉成 `(out_f * in_f / 128, 128)`，每一**行**就是上述的一个 group，`scale/zero` 按 `dim=1` 在该行内算，所以是「每 group 一套参数」。

---

## 1. ASD 的两部分（已实现）

- **公式**：`ASD = K×θ1 + Ψ×θ2`，对**每个输入特征（通道）**算一个 ASD；θ1、θ2 可调。
- **目标**：ASD 高的通道，在量化时保留对应**权重列**的精度。

**代码**：`asdq/metrics/asd.py`
- `compute_K(x, method)`：该通道“有多显著”（绝对），支持 mean/max/l2（K2/K1/K3）。
- `compute_Psi(K, method)`：该通道“相对其他通道有多显著”，支持 mean/max/zscore（Ψ1/Ψ2/Ψ3）。
- `compute_ASD(x, theta1, theta2, ...)`：归一化可选，返回 `[C]`，高 ASD → 保留该列精度。

---

## 2. K×θ1：绝对显著性（已实现）

- **含义**：这一路输入激活本身有多重要（如绝对值均值大 → 更显著）。
- 与常见通道级量化一致：用**激活**判断“这一通道是否重要”，再决定对应**权重列**精度。

**代码**：`compute_K` 的 `mean`（K2）、`max`（K1）、`l2`（K3）。预设见 `ASD_PRESETS`，yaml 中 `asd_preset` / `asd_k_method`。

---

## 3. Ψ×θ2：相对显著性（已实现）

- **含义**：同一层/同一组特征里，该特征相比其他是否“鹤立鸡群”（尖峰 → Ψ 高；多小峰 → Ψ 不会都高）。
- **K**：“这个通道自己强不强”；**Ψ**：“这个通道在众多通道里是否格外突出”。

**代码**：`compute_Psi(K, method)`，用同一层的 K 作为输入，得到相对显著性。

---

## 4. 混合精度共识（已实现）

### 4.1 原则

- 对每个输入通道算 ASD；**ASD 越高** → 该通道对应权重**列**越值得高精度（如 8bit）；越低 → 低精度（如 4bit）。

### 4.2 全局比例上限

- 设**全局** outlier 比例，例如 **0.1**（全模型最多 10% 的通道享受高精度）。
- 不是每层各自 10%，而是**所有层、所有通道一起**排序，只取前 10%。

**SpQR 风格共识（当前实现）**：分组保持「一行 × 一坨列」。组内落在「高精度列」上的权重视为 **outlier（保存精度）**。  
- **算量化参数**：组内先把 outlier 位置用该行「非保存位置」的**均值**填上，再按行算 scale/zero（方式 1，效仿 SpQR），即**只用「剩下没保留精度」的那部分**决定 scale/zero。  
- **量化与合并**：用该 scale/zero 量化整组，再用**原权重重写**保存精度位置；输出权重 = 反量化结果 × (1 − mask) + 原权重的 outlier 部分，即**推理时等价于「量化部分 + outlier 加回」**。  
- **某行某 group 全是保存列**：该行无「剩下的」可算；实现上用该行整段均值填充后仍算 scale/zero，最终用原权重重写整段，效果为该段不量化、整段当保存精度（方案 a）。

### 4.3 实现三步（代码对应）

| 步骤 | 共识 | 实现 |
|------|------|------|
| 第一步 | 对所有要量化的层，用校准激活算每通道 ASD → (层, 通道, ASD) | `collect_layer_activations` 收集激活；`compute_global_asd_list` 得到全局 ASD 列表 |
| 第二步 | 全模型 ASD 一起排序，取前 ratio 的通道标为「高精度」 | `select_high_precision_columns(global_asd_list, ratio)` |
| 第三步 | 按 group（一行×一坨列）剔除保存列后算 scale/zero，量化后合并 outlier | `pseudo_quantize_model_weight(..., high_precision_columns=..., ...)`，内部用 `pseudo_quantize_weight_spqr_style`：组内填均值 → `_get_scale_zero_per_row` → 量化后用原权重重写保存位置，得到合并权重 |

### 4.4 配置（可配置）

- **比例**：yaml 中 `asd_high_precision_ratio: 0.1`（可改）。
- **分组大小**：`w_group`（即 `q_group_size`，如 128），混合精度时一行×一坨列的大小。
- **低精度比特**：`asd_low_w_bit`（组内非保存列用量化比特）；保存精度列为原权重重写（outlier），不单独设高比特。
- **是否启用混合精度**：`asd_mixed_precision: true/false`。

### 4.5 推理

- 混合精度路径写回每层权重的已是 **合并后的权重**（量化反量化部分 + 保存精度位置的原值），因此**推理/前向无需改**：加载 `state_dict` 后直接做 matmul，等价于「outlier 的计算结果已加回」。

---

## 5. 使用方式

- **仅统一比特量化**（当前默认）：`asd_mixed_precision: false`，仅用 `w_bit` 与 `w_group`。
- **ASD 混合精度（SpQR 风格）**：在 yaml 中设置 `asd_mixed_precision: true`，并配置 `data_path`、`image_folder` 等校准数据；运行 `python main_quant.py --config configs/default.yaml` 时会：
  1. 用校准数据跑前向并收集每层 Linear 输入激活；
  2. 算全局 ASD 并取前 `asd_high_precision_ratio` 的通道；
  3. 按「一行×一坨列」分组，组内剔除保存精度列后用剩余位置算 scale/zero（填均值再 fit），量化后用原权重重写保存位置，得到合并权重并写回；推理时直接前向即可。

**层 key 约定**：校准与量化统一使用 `layers.{block_idx}.{linear_name}`，保证高精度集合与权重列一一对应。

---

## 6. 小结

- **K×θ1**：按激活的“绝对显著性”判断通道重要性（已实现，可配置方法）。
- **Ψ×θ2**：按“相对其他通道是否突出”判断（已实现，可配置方法）。
- **混合精度（SpQR 风格）**：分组 = 一行×一坨列；组内若有保存精度列则剔除（填均值）后算 scale/zero，量化后再用原权重重写保存位置；输出为合并权重，推理时直接前向即“outlier 加回”。

公式细节（是否严格按论文 K(A)=E[(A−μ)^4]/σ^4−3 等）可在 `asd.py` 中继续扩展或替换现有 K/Ψ 方法。
