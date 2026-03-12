# ASDQ

ASDQ: Activation-aware Statistical sensitivity driven Quantization for multimodal large models.

本仓库为多模态大模型量化方法 ASDQ 的代码实现，支持**运行量化（校准 + RTN 权重量化）**与**评估（Zeroshot / PPL）**全流程。

---

## 一、环境构建

与 [MBQ](https://github.com/ShiyaoLi/MBQ) 一致，建议使用同一 conda 环境。

```bash
# 1. 创建并激活 conda 环境（若与 MBQ 共用可跳过）
conda create -n asdq python=3.10 -y
conda activate asdq

# 2. 进入项目目录
cd E:\LLM-learning\ASDQ

# 3. 安装依赖
pip install -r requirements.txt
pip install -e .

# 4. 安装 lmms-eval 及 LLaVA 相关依赖（用于 llava_onevision）
# 若已按 MBQ 文档安装过，可跳过
pip install lmms-eval
# LLaVA-OneVision 需按官方仓库安装对应依赖
```

- **数据**：校准使用 COCO 格式（与 MBQ 相同），需自行准备 JSON/JSONL 与图像目录。
- **模型**：通过 lmms-eval 加载，需自行准备权重（如 `lmms-lab/llava-onevision-qwen2-7b-ov` 或本地路径）。

---

## 二、配置说明

所有可调参数（含 ASD 的 θ1/θ2、量化方案、评估任务）均在 **`configs/default.yaml`** 中，并配有中文注释，便于切换不同方案对比结果。

- **模型 / 校准**：`model`、`model_args`、`data_path`、`image_folder`、`n_samples` 等。
- **ASD**：`asd_preset`、`asd_k_method`、`asd_psi_method`、**`asd_theta1`**、**`asd_theta2`**（可选）、`asd_normalize`。
- **量化**：`run_process`、`scale_path`、`w_bit`、`w_group`、`pseudo_quant`。
- **评估**：`tasks`、`limit`、`output_path`、`seed`、`gen_kwargs` 等。

修改 yaml 后可直接用 `--config configs/default.yaml` 驱动量化与评估，无需改代码。

---

## 三、运行量化

先执行**校准 + 权重量化**，并将结果保存到 `scale_path`，供评估加载。

```bash
# 使用配置文件（推荐：在 configs/default.yaml 中填好 data_path、image_folder、model_args、scale_path）
python main_quant.py --config configs/default.yaml --run_process

# 或命令行覆盖部分项
python main_quant.py --config configs/default.yaml --run_process \
  --data_path /path/to/coco.json \
  --image_folder /path/to/coco/images \
  --model_args "pretrained=lmms-lab/llava-onevision-qwen2-7b-ov" \
  --scale_path scale_cache/asdq_llava_7b_w4.pt
```

- `--run_process`：执行校准与量化并写入 `scale_path`；不加则仅加载模型（例如仅做推理时不写盘可省略）。
- 量化完成后，`scale_path` 中为当前权重的 state_dict（伪量化后的权重），供下一步评估使用。

---

## 四、评估量化结果（Zeroshot / PPL）

使用 **lmms-eval** 做 Zeroshot 评估（及若任务支持则含 PPL）。

```bash
# 评估已量化模型：指定同一 config，并保证 scale_path 与量化时一致
python main_eval.py --config configs/default.yaml \
  --scale_path scale_cache/asdq_llava_7b_w4.pt \
  --tasks mmmu_val \
  --output_path eval_results

# 仅评估原始 FP 模型（不加载量化权重）
python main_eval.py --config configs/default.yaml \
  --tasks mmmu_val \
  --output_path eval_results
```

- **Zeroshot**：通过 `--tasks` 指定任务，如 `mmmu_val`、`mme`、`mmb` 等（与 lmms-eval 任务名一致）。可用 `--tasks list` 查看可用任务。
- **PPL**：若 lmms-eval 中某任务会汇报 perplexity，将该任务加入 `--tasks` 即可得到 PPL 结果；具体任务名请参考 lmms-eval 文档或 `--tasks list`。

结果会打印在终端，并若指定 `--output_path` 则写入该路径下的 `results.json`（或目录内的结果文件）。

---

## 五、全流程示例（从环境到评估）

```bash
# 1. 环境
conda activate asdq   # 或你的 MBQ 环境
cd E:\LLM-learning\ASDQ
pip install -r requirements.txt && pip install -e .

# 2. 在 configs/default.yaml 中设置：
#    model_args, data_path, image_folder, scale_path, tasks, output_path

# 3. 运行量化（校准 + RTN 权重量化，保存到 scale_path）
python main_quant.py --config configs/default.yaml --run_process

# 4. 评估量化后的模型（Zeroshot，可选 PPL）
python main_eval.py --config configs/default.yaml \
  --scale_path scale_cache/asdq_llava_7b_w4.pt \
  --tasks mmmu_val \
  --output_path eval_results
```

如需对比不同 ASD 方案或 θ1/θ2，只需修改 `configs/default.yaml` 中的 `asd_preset`、`asd_theta1`、`asd_theta2` 等，重新执行步骤 3 与 4 即可。

---

## 六、项目结构

- `main.py`：仅加载模型与校准数据（不写 scale_path，不跑量化）。
- `main_quant.py`：量化入口，校准 + RTN 量化 + 保存 state_dict 到 `scale_path`。
- `main_eval.py`：评估入口，Zeroshot（及支持 PPL 的任务），可加载 `scale_path` 评估量化模型。
- `configs/default.yaml`：全量配置项与中文注释。
- `docs/ARCHITECTURE.md`、`docs/ARCHITECTURE_DEEP_DIVE.md`：架构与各文件说明。
