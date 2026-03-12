# ASDQ 架构深度解析：文件作用、用法与设计原因

本文档按文件/模块逐项说明：**作用**、**用法**、**核心代码为什么这么写**，便于以后复用到类似项目（多模态模型 + 校准/量化管线）的构建中。

---

## 一、项目根目录

### 1. `main.py` — 入口脚本

**作用**  
- 唯一对外可执行入口：解析参数 → 加载模型 → 加载校准数据 → 当前仅占位（不执行量化）。  
- 把「配置」「模型加载」「数据加载」串成一条线，后续量化、评估都从这条线延伸。

**用法**  
```bash
# 命令行
python main.py --model llava_onevision --model_args "pretrained=xxx" --calib_data coco --data_path /path/to.json --image_folder /path/to/images --n_samples 128

# 用配置文件（推荐，便于复现）
python main.py --config configs/default.yaml
```

**核心代码为什么这么写**

| 写法 | 原因 |
|------|------|
| `parse_args()` 单独成函数 | 参数解析与业务逻辑分离；便于测试时传入 `Namespace`，也便于后续用 `--config` 覆盖。 |
| `config = [config] if not isinstance(config, list) else config` | 一份 yaml 可写单个配置（dict）或多组实验（list），入口统一按「配置列表」处理，便于批量跑多组参数。 |
| `args_copy = argparse.Namespace(**vars(args)); for k,v in cfg: setattr(args_copy, k, v)` | 先继承 CLI 默认值，再用 yaml 覆盖，这样「默认值在代码里、实验配置在 yaml 里」分工清晰。 |
| `get_model(args.model)` 来自 lmms-eval，再 `get_process_model(args.model)` 来自 asdq | 模型分两层：**评估/推理用**（lmms-eval 的 `lm`）和**校准/量化用**（asdq 的 process model）。入口只依赖两个 getter，不关心内部是 LLaVA 还是 Qwen2-VL，符合依赖倒置。 |
| `_run_single(args)` 与 `cli_main()` 分离 | 单次运行逻辑集中在 `_run_single`，`cli_main` 只做「有无 config、循环几次」的分发，后续加多 GPU、多 config 并行时只需改 `cli_main`。 |

---

### 2. `setup.py` — 包安装描述

**作用**  
- 让当前目录作为 Python 包被安装（`pip install -e .`），这样其他脚本里可以 `import asdq`。  
- 声明包名、版本、依赖（可在此扩展，当前依赖主要在 requirements.txt）。

**用法**  
```bash
cd E:\LLM-learning\ASDQ
pip install -e .
```

**核心代码为什么这么写**

| 写法 | 原因 |
|------|------|
| `setuptools.find_packages()` | 自动发现所有含 `__init__.py` 的子目录作为子包，无需手列 asdq、asdq.models 等，新增子包也不用改 setup.py。 |
| `long_description=open("README.md").read()` | PyPI 或本地 `pip show asdq` 会显示 README，方便他人了解项目。 |
| `encoding="utf-8"` | Windows 下默认编码可能不是 UTF-8，显式指定避免 README 含中文时安装报错。 |

---

### 3. `requirements.txt` 与 `configs/default.yaml`

**作用**  
- `requirements.txt`：固定依赖版本，保证「pip install -r requirements.txt」后环境一致。  
- `configs/default.yaml`：给出一份默认/示例配置，便于用 `--config configs/default.yaml` 跑通，再把路径改成自己的 data_path、image_folder。

**用法**  
- 安装依赖：`pip install -r requirements.txt`  
- 使用配置：在 yaml 里改 `data_path`、`image_folder`、`model_args` 后执行 `python main.py --config configs/default.yaml`

**设计原因**  
- 依赖与配置外置，避免写死在代码里；换机器、换数据只需改文本文件，符合 12-Factor 的「配置与代码分离」。

---

## 二、`asdq/` 主包

### 4. `asdq/__init__.py`

**作用**  
- 标记 `asdq` 为包，可被 `import asdq`。当前几乎为空，仅保留包身份；若以后想统一对外接口，可在这里写 `from asdq.xxx import yy`。

**用法**  
- 无需直接调用；安装包后 `import asdq` 或 `from asdq.models import get_process_model` 时会触发此文件。

---

### 5. `asdq/utils/registry.py` — 注册表

**作用**  
- 提供「名字 → 类/函数」的映射（如 `"llava_onevision"` → `LLaVA_onevision`），使主流程通过字符串选择实现，而不写死 if/else。

**用法**  

- **注册**：在具体模型类上写 `@MODEL_REGISTRY.register("llava_onevision")`，类加载时自动入表。  
- **查找**：`get_process_model("llava_onevision")` 内部用 `MODEL_REGISTRY["llava_onevision"]` 取到类再实例化。

**核心代码为什么这么写**

| 写法 | 原因 |
|------|------|
| `Register` 继承 `dict` 并重写 `__setitem__/__getitem__`，用 `self._dict` 存 | 对外表现像 dict（`REGISTRY["name"]`），但可以自定义 `register()` 的校验逻辑（如禁止重复 key、只允许 callable）。 |
| `register(target)` 支持两种用法：`@register` 装饰类，或 `register("name")(MyClass)` | 既可以用装饰器默认用类名当 key，也可以显式传字符串（如 `"llava_onevision"`），和 lmms-eval 的模型名一致，便于用同一字符串从两边取模型。 |
| `MODEL_REGISTRY`、`DATASET_REGISTRY`、`METHOD_REGISTRY` 三个表 | 模型、数据集、量化方法将来都可能有多实现，用同一套注册机制，扩展时只加新模块+装饰器，不改 main 或 calibration 的 if/else。 |

**可积累经验**：任何「按名字选实现」的场景（多后端、多模型、多策略），都可以用「注册表 + 装饰器」替代大量分支，符合开闭原则。

---

### 6. `asdq/models/base.py` — 抽象基类

**作用**  
- 定义所有「process model」必须实现的接口（`fetch_vit/llm/proj`、`vision_preprocess`、`preprocess_data`、`data_collator`、`generate_input` 等），校准与量化代码只依赖这些接口，不依赖具体 LLaVA/Qwen2-VL。

**用法**  

- 新模型适配时：新建 `asdq/models/xxx/`，写一个类继承 `BaseModel`，实现所有抽象方法，并用 `@MODEL_REGISTRY.register("xxx")` 注册；无需改 calibration 或 main。

**核心代码为什么这么写**

| 写法 | 原因 |
|------|------|
| 用 `@abc.abstractmethod` | 子类若漏实现某个方法，实例化时就会报错，而不是在运行到某一步才崩，便于早期发现接口不一致。 |
| `fetch_vit`、`fetch_llm`、`fetch_proj` | 量化时经常要按「视觉编码器 / LLM / 投影层」分别处理（如只量化 LLM、或对 proj 做特殊缩放），抽象成三个 getter 让量化层能统一拿到子模块。 |
| `__call__` 转发到 `forward` | 使 `process_model(...)` 与 `process_model.forward(...)` 等价，和 PyTorch 的 `nn.Module` 习惯一致，量化代码里写 `model(inputs)` 即可。 |

**可积累经验**：多实现、多后端时，先抽「最小必要接口」的基类，再让各实现填满接口；调用方只依赖基类，扩展时只加新子类。

---

### 7. `asdq/models/__init__.py` — 模型入口

**作用**  

- 提供 `get_process_model(model_name)`；  
- 通过 import 触发各子模块的 `@MODEL_REGISTRY.register(...)`，把名字注册进表。

**用法**  
```python
from asdq.models import get_process_model
ProcessModelClass = get_process_model("llava_onevision")
process_model = ProcessModelClass(lm._model, lm._tokenizer, lm.processor)
```

**核心代码为什么这么写**

| 写法 | 原因 |
|------|------|
| `from asdq.models.llava_onevision import LLaVA_onevision  # noqa: F401` | 目的不是用 `LLaVA_onevision` 这个变量，而是执行该模块顶层代码，从而执行 `@MODEL_REGISTRY.register("llava_onevision")`；F401 表示「未使用的 import」，用 noqa 避免 lint 报错。 |
| `get_process_model` 只做 `return MODEL_REGISTRY[model_name]` | 入口极薄，所有「如何构造实例」的逻辑都在各模型类的 `__init__` 里，这里只负责按名字返回类，便于测试和替换。 |

**可积累经验**：用「import 副作用」做注册时，在 `__init__.py` 里集中 import 一遍即可，调用方只需 `get_xxx(name)`，无需知道具体类在哪个文件。

---

### 8. `asdq/models/llava_onevision/llava_onevision.py` — LLaVA 适配器

**作用**  
- 把 lmms-eval 给出的「原始 LLaVA 模型 + tokenizer + processor」包装成满足 BaseModel 的 process model，供校准和后续量化使用。  
- 实现 LLaVA 特有的多模态预处理、padding、图像 token 替换、`prepare_inputs_labels_for_multimodal` 等。

**用法**  
- 由 main 通过 `get_process_model("llava_onevision")` 间接使用；一般不直接实例化。

**核心代码为什么这么写**

| 写法 | 原因 |
|------|------|
| `__init__` 里保存 `model/tokenizer`，并取出 `vision_tower`、`image_processor`、`mm_use_im_start_end` 等 | 校准与量化时频繁需要「按层/按模块」访问，在初始化时抽成属性，避免到处写 `model.model.vision_tower` 这种长路径，也便于以后做设备迁移（如 to_cuda）。 |
| `vision_preprocess` 返回 `(tensor, image_size, "image")` 三元组 | 多模态模型需要区分「这一段是图像 embedding 还是文本」；用第三个元素 `"image"`/`"text"` 标记，后面 `data_collator`、`generate_input` 里可以按 modality 拼 batch 或算 mask。 |
| `preprocess_data(images, data_item)`：先处理图像与对话格式，再调 `self.preprocess(sources, tokenizer, has_image)` | 校准数据是「原始 JSON 的一条」；这里负责把一条变成模型需要的 `input_ids/labels/image` 等。不同对话模板（v1、llama2、qwen）在 `dataset.py` 的 `preprocess_*` 里，这里用 `self.preprocess` 统一入口，根据 `conversation_lib.default_conversation` 选具体实现。 |
| `data_collator(instances)`：pad 到同一长度、拼成 batch，并保留 `image_sizes`、`modalities`、`images` | 校准时通常一次喂一个 batch；collator 负责「多条样本 → 一个 batch dict」，且多模态需要把 image 列表和 modality 信息一起传给 `generate_input`，所以不能只拼 `input_ids/labels`。 |
| `generate_input(data_samples)`：调 `prepare_inputs_labels_for_multimodal`，再算 `vision_sel`、`vision_mask`、`answer_mask` | 量化/校准时需要的是「已经变成 embedding 的输入」和「哪里是图像、哪里要算损失」的 mask；这里把 LLaVA 的 prepare 逻辑封装好，返回 `prompt_inputs`（如 inputs_embeds）和 `prompt_kwargs`（labels、attention_mask、vision_mask 等），后续量化层只消费这两个，不关心 LLaVA 内部细节。 |
| `@torch.no_grad()` 在 `generate_input`、`few_shot_data_samples` 等上 | 校准阶段不反传，关掉梯度可省显存、避免误更新参数。 |

**可积累经验**：为某一框架（如 LLaVA）做「校准/量化适配」时，核心就是实现「单条样本 → preprocess_data」「多条 → data_collator」「batch dict → generate_input 得到模型真正吃的输入和 mask」这三步，且与 BaseModel 约定好入参和返回值形状，上层就能通用。

---

### 9. `asdq/models/llava_onevision/dataset.py` — 对话与 tokenization

**作用**  
- 提供多种对话模板下的 `preprocess_*`（如 `preprocess_llama_2`、`preprocess_v1`、`preprocess_qwen` 等），把「对话列表」转成 `input_ids` 和 `labels`（含 IGNORE_INDEX 的 mask）。  
- 被 `llava_onevision.py` 的 `preprocess()` 调用，不直接被 main 或 calibration 使用。

**用法**  
- 仅作为 LLaVA 适配器的依赖；若新增一种对话格式，在此文件加新的 `preprocess_xxx` 并在 `preprocess()` 里加分支。

**核心代码为什么这么写**

| 写法 | 原因 |
|------|------|
| 各 `preprocess_*` 接收 `sources`（list of conversation）、`tokenizer`、`has_image` | 不同 LLaVA 版本/底座（Llama2、Qwen、Gemma）的模板不同，拆成多个函数便于维护；`has_image` 决定是否用 `tokenizer_image_token` 处理 `<image>`。 |
| `labels` 里用 `IGNORE_INDEX` 遮住「不需要算损失」的位置（如 system、user） | 多模态校准/量化时通常只对「模型要生成的那一段」算损失，mask 掉其他部分，和训练时的做法一致。 |
| 与 `llava` 库的 `conversation_lib`、`tokenizer_image_token` 等强耦合 | 这是「单模型适配」的实现细节，故意不抽象到 base，避免 base 依赖具体框架；其他模型（如 Qwen2-VL）会有自己的 dataset 模块。 |

---

### 10. `asdq/calibration/coco_vl.py` — 多模态校准数据

**作用**  
- 从 JSON/JSONL 读样本，按 `image_folder` 拼图像路径，调用 process model 的 `preprocess_data` → `data_collator` →（可选）`few_shot_data_samples` / `interleave_data_samples` → `generate_input`，得到 `(prompt_inputs, prompt_kwargs)` 供量化使用。

**用法**  
```python
from asdq.calibration.coco_vl import get_multimodal_calib_dataset
prompt_inputs, prompt_kwargs = get_multimodal_calib_dataset(
    data_path="/path/to/coco.json",
    image_folder="/path/to/images",
    model=process_model,
    n_samples=128,
)
```

**核心代码为什么这么写**

| 写法 | 原因 |
|------|------|
| 支持 `.json` 与 `.jsonl` 两种读法 | 不同数据管线产出格式不同，入口统一成「列表 of dict」，后面逻辑一致。 |
| `shuffle` 用固定 seed（如 42） | 校准可复现；换 seed 可做简单敏感性检查。 |
| `for i in range(n_samples): i = i % len(dataset)` | 样本数不足 n_samples 时循环使用，避免报错；若数据足够大，效果等价于取前 n_samples。 |
| 每条样本调用 `model.preprocess_data(images, data_item)`，再 `model.data_collator(data_list)` | **不写死**图像尺寸、对话格式、tokenizer；全部委托给 process model，这样换模型只需换 process model，校准代码不动，符合「面向接口编程」。 |
| `few_shot_format` / `interleave_format` 在 collate 之后、`generate_input` 之前做 | 这些是「batch 级」的再组织（如多图+多轮对话、图文交错），仍用 model 的接口（`few_shot_data_samples`、`interleave_data_samples`），校准层只负责开关和传参。 |
| 最后统一 `model.generate_input(examples)` | 无论是否 few_shot/interleave，量化需要的都是同一形式的 `(prompt_inputs, prompt_kwargs)`，所以入口只在一处调 `generate_input`，逻辑清晰。 |

**可积累经验**：校准数据层只做「读文件 → 按条预处理 → 拼 batch → 转成模型输入」；「怎么预处理、怎么拼」都交给 model 的接口，这样换数据集格式时主要改读文件部分，换模型时只换 model，职责清晰。

---

## 三、整体设计可复用的几条原则

1. **入口薄、逻辑在包内**：main 只做「解析 → 调 get_model / get_process_model / get_multimodal_calib_dataset」，具体实现都在 asdq 各子模块。  
2. **按名字选实现**：用注册表 + 装饰器，避免 main 或 calibration 里堆 if/else。  
3. **抽象接口在 base，具体在子类**：校准和量化只依赖 BaseModel 的接口，新模型只需实现接口并注册。  
4. **配置与代码分离**：requirements.txt、config yaml 外置，方便换环境和复现。  
5. **数据流单向**：main → 模型加载 → process model → 校准数据 → 得到 (prompt_inputs, prompt_kwargs)；后续量化也只消费这两个，不回头改数据加载。

按上述方式组织，以后做「新模型适配」或「新校准数据源」时，只需加新文件/新类并注册，无需改主流程，便于积累成可长期维护的项目结构。
