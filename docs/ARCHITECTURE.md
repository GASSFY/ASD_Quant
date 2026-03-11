# ASDQ 架构说明

本文档说明当前 ASDQ 项目的基础架构：各目录与模块的职责，以及这样划分的原因。

---

## 一、整体结构

```
ASDQ/
├── main.py                 # 入口：解析参数 → 加载模型 → 加载 COCO 校准数据 → 占位（暂不量化）
├── configs/
│   └── default.yaml        # 默认配置（model、data_path、image_folder 等）
├── asdq/                   # 主包
│   ├── calibration/       # 校准数据加载
│   ├── models/            # 模型适配层（process model）
│   └── utils/              # 注册表等工具
├── requirements.txt
├── setup.py
└── docs/
    └── ARCHITECTURE.md    # 本文件
```

---

## 二、各部分作用与设计理由

### 1. `main.py`（入口）

- **作用**：解析命令行/配置文件 → 用 lmms-eval 加载原始模型 → 用 asdq 的 `get_process_model` 得到「处理模型」→ 用 COCO 校准接口加载 `(prompt_inputs, prompt_kwargs)` → 当前仅打印占位信息，不执行量化。
- **为何单独入口**：与 MBQ 一致，量化/评估入口与库代码分离，便于用 `--config` 批量跑不同配置，也便于后续在此处接 ASD 度量、DIRECT、敏感层划分和量化流程。

### 2. `configs/`（配置）

- **作用**：用 YAML 保存默认或常用配置（如 `model`、`model_args`、`data_path`、`image_folder`、`n_samples` 等），`main.py` 通过 `--config` 读取并覆盖 CLI 参数。
- **为何要配置目录**：与 MBQ 一致，方便按模型/实验维护多份配置，避免长命令行；后续可为不同模型或量化设置增加更多 yaml。

### 3. `asdq/`（主包）

- **作用**：提供「模型适配 + 校准数据」能力，不直接依赖 lmms-eval 的 API，而是通过统一的 process model 接口与校准、后续量化交互。
- **为何单独成包**：便于 `pip install -e .` 安装，被 `main.py` 和后续脚本以 `import asdq...` 方式使用；与 MBQ 的 qmllm 包对应，便于对照和迁移。

### 4. `asdq/calibration/`（校准数据）

- **作用**：目前仅实现 `get_multimodal_calib_dataset()`，从 JSON/JSONL 读样本、按 `image_folder` 拼图像路径、调用 process model 的 `preprocess_data` → `data_collator` → `generate_input`，返回 `(prompt_inputs, prompt_kwargs)`。
- **为何独立成 calibration**：校准数据格式（COCO/多模态）与「如何用模型预处理」强相关，但与「具体是哪种 VLM」解耦：只要 process model 实现统一接口即可。后续若增加 pileval 等纯文本校准，可在此包内加新函数，与 MBQ 一致。

### 5. `asdq/models/`（模型适配层）

- **作用**：
  - **base.py**：定义 `BaseModel` 抽象基类（如 `fetch_vit`、`fetch_llm`、`vision_preprocess`、`preprocess_data`、`data_collator`、`generate_input` 等），保证所有 VLM 适配器行为一致。
  - **llava_onevision/**：实现 LLaVA-OneVision 的 process model，封装 `lm._model`、tokenizer、processor，提供校准与后续量化所需的输入格式。
  - **__init__.py**：通过 `get_process_model(model_name)` 从 `MODEL_REGISTRY` 返回对应适配类；当前仅注册 `llava_onevision`。
- **为何要 process model 层**：lmms-eval 给出的 `lm` 面向评估，而校准与量化需要统一的多模态输入构造（图像+文本、padding、embedding 准备等）。process model 把「模型 + tokenizer + processor」包成统一接口，方便 calibration 和后续量化模块只依赖这一层，而不关心底层是 LLaVA 还是 Qwen2-VL 等。与 MBQ 设计一致，便于后续扩展 InternVL2、Qwen2-VL 等。

### 6. `asdq/utils/`（工具与注册表）

- **作用**：`registry.py` 提供 `MODEL_REGISTRY`（以及 `DATASET_REGISTRY`、`METHOD_REGISTRY` 占位），用于按字符串名注册/获取 process model 类。
- **为何用注册表**：新增模型时只需在新模块里 `@MODEL_REGISTRY.register("xxx")` 并在 `models/__init__.py` 中 import，无需改 `main.py` 或 calibration 逻辑，符合开闭原则；与 MBQ 一致。

---

## 三、数据流（当前阶段）

1. **main.py** 解析参数（含可选的 `--config`）。
2. 使用 **lmms-eval** 的 `get_model(args.model)` 创建 `lm`（含 `lm._model`、`lm._tokenizer`、`lm.processor`）。
3. 使用 **asdq** 的 `get_process_model(args.model)` 得到 process model 类并实例化 `process_model`。
4. 若 `calib_data=coco` 且提供了 `data_path`、`image_folder`，则调用 **asdq.calibration** 的 `get_multimodal_calib_dataset(..., model=process_model)`，得到 `(prompt_inputs, prompt_kwargs)`。
5. 当前仅打印「模型与校准数据已加载，量化未实现」；后续将在此处接入 ASD 度量、敏感层划分、DIRECT 与量化流程。

---

## 四、后续可扩展点

- **asdq/quantization/**：量化算子、ASD 公式、敏感层划分、DIRECT 内外层优化。
- **asdq/methods/** 或 **asdq/quantization/**：按方法（如 asdq、rtn）组织具体量化逻辑。
- **asdq/models/**：新增 `qwen2_vl`、`internvl2` 等目录，实现对应 BaseModel 并注册，即可复用同一套 calibration 与 main 流程。
