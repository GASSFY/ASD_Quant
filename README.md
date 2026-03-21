# ASDQ

**ASDQ**: Activation-aware Significance-driven Quantization for multimodal large models.

基于 Hessian 对角线的通道重要性排序与分组混合精度量化，支持**校准 → ASD 排序 → 量化 → 评估**全流程。

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

# 4. 安装LLAVA-NEXT依赖
cd /root/autodl-tmp
git clone https://github.com/LLaVA-VL/LLaVA-NeXT.git
cd LLaVA-NeXT
pip install -e .

# 5. 安装lmms-eval依赖
git clone https://github.com/LSY-noya/lmms-eval.git
cd lmms-eval
pip install -e .
```

### 数据与模型准备

- **校准数据**：COCO 格式的 JSON/JSONL + 图像目录
- **模型权重**：`lmms-lab/llava-onevision-qwen2-7b-ov` 或本地路径

---

## 二、YAML 全参数说明

所有可调参数在 `configs/default.yaml` 中。下表按逻辑分组逐项说明；**适用脚本**一列注明该参数主要被 `main_quant.py`（量化）、`main_eval.py`（评估）或两者共用。

| 分组 | 参数 | 默认值 | 说明 | 适用脚本 |
|------|------|--------|------|----------|
| **模型** | `model` | `llava_onevision` | 模型类型，与命令行 `--model` 对应 | 共用 |
| | `model_args` | `"pretrained=your/model-name"` | lmms-eval 风格字符串，如 `pretrained=xxx,dtype=float16`，用于加载权重与设备 | 共用 |
| | `batch_size` | `"1"` | 量化/评估时的 batch 大小（字符串） | 共用 |
| | `device` | `null` | 设备，如 `cuda:0`；`null` 表示自动 | 共用 |
| **校准数据** | `calib_data` | `coco` | 校准数据源类型，目前仅支持 `coco` | main_quant |
| | `n_samples` | `128` | 校准样本数，影响 Hessian 估计质量与耗时 | main_quant |
| | `data_path` | `""` | COCO 格式 JSON/JSONL 路径（校准用） | main_quant |
| | `image_folder` | `""` | 校准图像所在目录 | main_quant |
| | `interleave_format` | `false` | 是否使用交错格式加载校准数据 | main_quant |
| | `few_shot_format` | `false` | 是否使用 few-shot 格式 | main_quant |
| | `text_data_path` | `""` | 可选纯文本校准数据路径 | main_quant |
| **ASD** | `asd_theta1` | `0.8` | K（绝对显著性）的权重 | main_quant |
| | `asd_theta2` | `0.2` | Psi（相对显著性）的权重 | main_quant |
| **混合精度** | `asd_mixed_precision` | `true` | 是否启用 ASD 混合精度；为 true 时需提供校准数据 | main_quant |
| | `asd_high_precision_ratio` | `0.1` | 全局保留 float 的通道比例（全层一起排序） | main_quant |
| | `asd_low_w_bit` | `4` | 非高精度列的量化比特数 | main_quant |
| **量化** | `run_process` | `true` | `true`：执行校准+量化并保存；`false`：仅从 `scale_path` 加载已有量化权重，不跑校准与量化 | main_quant |
| | `scale_path` | `"scale_cache/asdq_llava_7b_w4.pt"` | 量化 state_dict 保存/加载路径（.pt）；评估时不传则跑原始 FP 模型 | 共用 |
| | `w_bit` | `4` | 未启用混合精度时的统一权重量化比特 | main_quant |
| | `w_group` | `128` | 分组大小：一行×w_group 列为一组，每组独立 scale/zero | main_quant |
| | `pseudo_quant` | `true` | 是否对权重做伪量化 | main_quant |
| **评估** | `tasks` | `"mmmu_val"` | lmms-eval 任务名，逗号分隔，如 `mmmu_val` 或 `mme,mmb` | main_eval |
| | `limit` | `null` | 每任务样本数上限（调试用），`null` 表示不限制 | main_eval |
| | `output_path` | `"eval_results"` | 评估结果目录或文件路径，结果写 results.json | main_eval |
| | `seed` | `"0,1234,1234,1234"` | 随机种子（四元组字符串） | main_eval |
| | `gen_kwargs` | `""` | 生成参数，传给 lmms-eval | main_eval |
| | `log_samples` | `false` | 是否记录每条样本的预测（调试用） | main_eval |
| | `verbosity` | `"INFO"` | 日志级别 | main_eval |
| | `results_md` | （未在 default 中） | 若设置，评估结果会追加到该 Markdown 文件（如消融脚本所用） | main_eval |

修改 yaml 后使用 `--config configs/default.yaml` 即可驱动量化或评估，无需改代码。算法细节见 [docs/ASD_AND_MIXED_PRECISION_CONSENSUS.md](docs/ASD_AND_MIXED_PRECISION_CONSENSUS.md)。

**参数来源规则**：使用 `--config` 时，**所有参数以配置文件为准**，命令行不会覆盖；**不使用** `--config` 时，所有参数需在命令行指定（见下方示例）。

---

## 三、运行量化

在 `configs/default.yaml` 中填好 `model_args`、`data_path`、`image_folder`、`scale_path` 等，然后：

```bash
python main_quant.py --config configs/default.yaml
```

仅加载已有量化权重（不校准、不重新量化）：在 yaml 中设 `run_process: false` 并设好 `scale_path`，再执行上述命令。

不用 config、全用命令行时，需传入所有必要参数，例如：

```bash
python main_quant.py --model llava_onevision --model_args "pretrained=MODEL,dtype=float16" --data_path /path/to/coco.json --image_folder /path/to/images --scale_path scale_cache/out.pt --n_samples 128
```

量化流程：加载模型与校准数据 → 流式收集 Hessian 对角 → 计算 importance 与 ASD → 全局排序选高精度列 → 分组混合精度量化 → 合并权重保存到 `scale_path`。

---

## 四、评估

在 yaml 中设好 `tasks`、`output_path`；评估量化模型时设好 `scale_path`，评估 FP16 时则不设或令其指向不存在的文件。然后：

```bash
# 评估量化后的模型（yaml 中 tasks、scale_path、output_path 已配置）
python main_eval.py --config configs/default.yaml

# 评估 FP16（yaml 中 scale_path 不设或对应文件不存在）
python main_eval.py --config configs/default.yaml
```

不用 config 时，所有参数在命令行指定，例如：

```bash
python main_eval.py --model llava_onevision --model_args "pretrained=MODEL,dtype=float16" --tasks mmmu_val --output_path eval_results
# 评估量化模型时再加上：--scale_path scale_cache/asdq_llava_7b_w4.pt
```

列出可用任务可调用 lmms-eval 的 `TaskManager.list_all_tasks()` 或查 lmms-eval 文档。

---

### 消融（run_ablation.sh）

在项目根目录执行：`bash run_ablation.sh`。脚本会按配置的任务列表（默认 mmmu_val、realworldqa、ocrbench、ai2d）逐任务执行：写 config 的 `tasks` → FP16 基线评估 → 多组 theta1/theta2/ratio 消融（改 yaml → 量化 → 评估 → 结果追加到 ablation_results.md → 删除当前 .pt）。

---

## 五、项目结构

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
│   │   ├── quant_funcs.py                 # 分组混合精度量化原语
│   │   └── mixed_precision.py             # 全局 ASD 排序与高精度列选择
│   └── utils/
│       └── registry.py                    # 注册表
├── docs/
│   └── ASD_AND_MIXED_PRECISION_CONSENSUS.md  # 设计共识文档
├── requirements.txt
└── setup.py
```

---

## 六、核心算法简述

**ASD 公式**：`ASD_c = θ1 × K_normalized + θ2 × Psi_normalized`

- **K**（绝对显著性）= `importance_c = ||W[:, c]||² × E[x_c²]`，度量量化第 c 列权重对输出方差的影响，跨层可比
- **Psi**（相对显著性）= importance 在层内的 z-score，度量该通道在同层中的异常程度
- K 全局归一化到 [0,1]，Psi 全局归一化到 [0,1]，加权合并后全局排序，取 top ratio 保留原始精度

**量化方式**：分组混合精度 — 按一行×w_group 列分组，每组独立 scale/zero；组内高精度列先用非高精度位置均值填充后拟合 scale/zero，量化后再写回原始权重，推理时直接 matmul。详见 [docs/ASD_AND_MIXED_PRECISION_CONSENSUS.md](docs/ASD_AND_MIXED_PRECISION_CONSENSUS.md)。
