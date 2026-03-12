# ASDQ

ASDQ: ASD Quantization for multimodal large models.

本仓库为多模态大模型量化方法 ASDQ 的代码实现，当前为**基础架构阶段**：支持加载模型与 COCO 校准数据，量化流程后续迭代加入。

## 环境

与 [MBQ](https://github.com/ShiyaoLi/MBQ) 一致，建议使用同一 conda 环境：

- Python 3.8+
- 安装依赖：`pip install -r requirements.txt`
- 安装本包：`pip install -e .`
- 需单独安装 lmms-eval 及 LLaVA 相关依赖（用于 `llava_onevision`）

## 数据与模型

- 校准数据：COCO 格式（与 MBQ 相同），需自行准备 JSON/JSONL 及图像目录。
- 模型：通过 lmms-eval 加载，需自行准备权重与配置。

## 运行

```bash
python main.py --model llava_onevision --model_args "pretrained=..." --calib_data coco --data_path /path/to/coco.json --image_folder /path/to/images --n_samples 128
```

当前仅完成「加载模型 + 加载 COCO 校准数据」流程，不执行量化。

## 项目结构

见仓库内 `docs/ARCHITECTURE.md`（构建完成后生成）。
