# ASDQ

**ASDQ**: Activation-aware Significance-driven Quantization for multimodal large models.

基于 Hessian 对角线的通道重要性排序 + SpQR 风格混合精度量化，支持**校准 → ASD 排序 → 量化 → 评估**全流程。

---

## 一、环境构建

```bash
# 1. 创建并激活 conda 环境
conda create -n asdq python=3.10 -y
conda activate asdq

# 2. 进入项目目录
cd E:\LLM-learning\ASDQ

# 3. 安装依赖
pip install -r requirements.txt
pip install -e .

# 4. 安装 lmms-eval（用于模型加载与评估）
pip install lmms-eval
```
# 还有下面这段
cd /root/autodl-tmp
git clone https://github.com/LLaVA-VL/LLaVA-NeXT.git
cd LLaVA-NeXT
pip install -e .

git clone https://github.com/LSY-noya/lmms-eval.git
cd lmms-eval
pip install -e .

### 数据与模型准备

- **校准数据**：COCO 格式的 JSON/JSONL + 图像目录
- **模型权重**：通过 lmms-eval 加载，如 `lmms-lab/llava-onevision-qwen2-7b-ov` 或本地路径

---

## 二、配置说明

所有可调参数在 `configs/default.yaml` 中，主要分为四组：

### ASD 显著性参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `asd_theta1` | `0.8` | K（绝对显著性）的权重 |
| `asd_theta2` | `0.2` | Psi（相对显著性/层内 z-score）的权重 |
| `asd_normalize` | `true` | 是否将 K/Psi 归一化到 [0,1] 再合并 |

### 混合精度参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `asd_mixed_precision` | `true` | 是否启用 ASD 混合精度 |
| `asd_high_precision_ratio` | `0.1` | 全局保留原始精度的通道比例（所有层一起排序） |
| `asd_low_w_bit` | `4` | 非 outlier 通道的量化比特数 |

### 量化参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `w_bit` | `4` | 统一量化比特数（未启用混合精度时使用） |
| `w_group` | `128` | SpQR 分组大小（一行 × w_group 列为一组） |
| `scale_path` | `"scale_cache/asdq_llava_7b_w4.pt"` | 量化权重保存/加载路径 |

### 评估参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `tasks` | `"mmmu_val"` | lmms-eval 任务名，逗号分隔 |
| `output_path` | `"eval_results"` | 评估结果输出目录 |
| `log_samples` | `false` | 是否记录每条样本的预测（调试用） |

修改 yaml 后用 `--config configs/default.yaml` 即可驱动全流程，无需改代码。

---

## 三、运行量化

**前提**：在 `configs/default.yaml` 中填写好 `model_args`、`data_path`、`image_folder`、`scale_path`。

```bash
# 使用配置文件运行（推荐）
python main_quant.py --config configs/default.yaml

# 或命令行覆盖部分配置
python main_quant.py --config configs/default.yaml \
  --data_path /path/to/coco.json \
  --image_folder /path/to/coco/images \
  --model_args "pretrained=lmms-lab/llava-onevision-qwen2-7b-ov,dtype=float16" \
  --scale_path scale_cache/asdq_llava_7b_w4.pt
```

量化流程会自动执行：
1. 加载模型与校准数据
2. 前向推理，流式收集 Hessian 对角线 `diag(H)_c = E[x_c²]`
3. 计算 `importance = ||W[:, c]||² × diag(H)_c`，全局 ASD 排序，选出 top ratio 高精度列
4. SpQR 风格分组量化（outlier 列保留原始精度）
5. 合并权重保存到 `scale_path`

---

## 四、评估量化结果

```bash
# 评估量化后的模型
python main_eval.py --config configs/default.yaml \
  --scale_path scale_cache/asdq_llava_7b_w4.pt \
  --tasks mmmu_val \
  --output_path eval_results

# 评估原始 FP 模型（不加载量化权重，去掉 --scale_path）
python main_eval.py --config configs/default.yaml \
  --tasks mmmu_val \
  --output_path eval_results

# 多任务评估
python main_eval.py --config configs/default.yaml \
  --scale_path scale_cache/asdq_llava_7b_w4.pt \
  --tasks mmmu_val,mme,mmb \
  --output_path eval_results
```

---

## 五、全流程示例

```bash
# 1. 环境准备
conda activate asdq
cd E:\LLM-learning\ASDQ
pip install -r requirements.txt && pip install -e .

# 2. 编辑 configs/default.yaml，填写：
#    model_args: "pretrained=lmms-lab/llava-onevision-qwen2-7b-ov,dtype=float16"
#    data_path: "/path/to/coco_calib.json"
#    image_folder: "/path/to/coco/images"
#    scale_path: "scale_cache/asdq_llava_7b_w4.pt"

# 3. 量化（校准 + ASD 排序 + SpQR 量化 + 保存）
python main_quant.py --config configs/default.yaml

# 4. 评估
python main_eval.py --config configs/default.yaml \
  --scale_path scale_cache/asdq_llava_7b_w4.pt \
  --tasks mmmu_val \
  --output_path eval_results
```

---

## 六、项目结构

```
ASDQ/
├── main_quant.py                          # 量化入口
├── main_eval.py                           # 评估入口
├── configs/
│   └── default.yaml                       # 全量配置
├── asdq/
│   ├── calibration/
│   │   ├── hessian_collector.py           # 流式 Hessian 对角收集
│   │   └── coco_vl.py                     # COCO 校准数据加载
│   ├── metrics/
│   │   └── asd.py                         # importance / Psi / ASD 计算
│   ├── models/
│   │   ├── base.py                        # BaseModel 抽象基类
│   │   └── llava_onevision/               # LLaVA-OneVision 适配器
│   ├── quantization/
│   │   ├── quantize.py                    # 模型级伪量化入口
│   │   ├── quant_funcs.py                 # SpQR 风格量化原语
│   │   └── mixed_precision.py             # 全局 ASD 排序与高精度列选择
│   └── utils/
│       └── registry.py                    # 注册表
├── docs/
│   └── ASD_AND_MIXED_PRECISION_CONSENSUS.md  # 设计共识文档
├── requirements.txt
└── setup.py
```

---

## 七、核心算法简述

**ASD 公式**：`ASD_c = θ1 × K_normalized + θ2 × Psi_normalized`

- **K**（绝对显著性）= `importance_c = ||W[:, c]||² × E[x_c²]`，度量量化第 c 列权重对输出方差的影响，跨层可比
- **Psi**（相对显著性）= importance 在层内的 z-score，度量该通道在同层中的异常程度
- K 全局归一化到 [0,1]，Psi 全局归一化到 [0,1]，加权合并后全局排序，取 top ratio 保留原始精度

**量化方式**：SpQR 风格混合精度 — 分组（一行 × w_group 列），组内 outlier 列用非 outlier 均值替代后算 scale/zero，量化后写回原始权重。推理时直接 matmul。
