"""
Minimal InternVL2 inference comparison script.

Only one model is supported: OpenGVLab/InternVL2-8B.
The script only does:
1) load model (FP16 or FP16+checkpoint)
2) process input
3) generate output
4) save records
"""
import argparse
import json
import os
from datetime import datetime
from typing import Any

import torch
from PIL import Image

from lmms_eval.models import get_model

from asdq.models import get_process_model

MODEL_NAME = "internvl2"
MODEL_ARGS = "pretrained=OpenGVLab/InternVL2-8B"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="InternVL2 FP16/Checkpoint generation compare")
    parser.add_argument("--prompt", type=str, default=None, help="Single prompt input.")
    parser.add_argument("--image", type=str, default=None, help="Optional image path for single prompt.")

    parser.add_argument("--scale_path", type=str, default=None, help="Optional quantized .pt checkpoint path.")
    parser.add_argument("--fp16_only", action="store_true", default=False, help="Only run FP16 model.")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--debug", action="store_true", default=False, help="Print generation debug info.")

    parser.add_argument("--batch_size", type=str, default="1")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output_json", type=str, default="outputs/prompt_compare.json")
    parser.add_argument("--output_md", type=str, default="outputs/prompt_compare.md")
    return parser.parse_args()


def _build_samples(args: argparse.Namespace) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if args.prompt:
        out.append({"id": f"cli_{len(out)}", "prompt": args.prompt, "image": args.image})
    if not out:
        raise ValueError("Please provide --prompt.")
    return out


def _load_internvl2(batch_size: str, device: str | None, scale_path: str | None):
    model_class = get_model(MODEL_NAME)
    lm = model_class.create_from_arg_string(
        MODEL_ARGS,
        {"batch_size": batch_size, "device": device},
    )
    if scale_path:
        if not os.path.exists(scale_path):
            raise FileNotFoundError(f"Checkpoint not found: {scale_path}")
        state = torch.load(scale_path, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "state_dict" in state:
            lm._model.load_state_dict(state["state_dict"], strict=False)
        else:
            lm._model.load_state_dict(state, strict=False)

    process_model = get_process_model(MODEL_NAME)(
        lm._model,
        lm._tokenizer,
        lm.processor if hasattr(lm, "processor") else None,
    )
    if hasattr(process_model, "to_cuda"):
        process_model.to_cuda()
    return lm, process_model


def _build_data_item(prompt: str, has_image: bool, sample_id: str) -> dict[str, Any]:
    data_item: dict[str, Any] = {
        "id": sample_id,
        "conversations": [
            {"from": "human", "value": prompt},
            {"from": "gpt", "value": ""},
        ],
    }
    if has_image:
        data_item["image"] = "placeholder"
    return data_item


def _generate_text(process_model, sample: dict[str, Any], max_new_tokens: int, debug: bool = False) -> str:
    prompt = str(sample["prompt"]).strip()
    if not prompt:
        raise ValueError(f"Sample {sample.get('id', 'unknown')} has empty prompt.")

    image_path = sample.get("image")
    images = None
    if image_path:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        images = [Image.open(image_path).convert("RGB")]

    data_item = _build_data_item(prompt, images is not None, str(sample.get("id", "sample")))
    processed = process_model.preprocess_data(images, data_item)
    batch = process_model.data_collator([processed])
    forward_kwargs, _ = process_model.generate_input(batch)

    llm = process_model.fetch_llm()
    tokenizer = process_model.tokenizer
    with torch.inference_mode():
        out_ids = llm.generate(
            inputs_embeds=forward_kwargs["inputs_embeds"],
            attention_mask=forward_kwargs["attention_mask"],
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    prompt_len = int(forward_kwargs["attention_mask"][0].sum().item())
    out_len = int(out_ids.shape[-1])
    raw_decode = tokenizer.decode(out_ids[0], skip_special_tokens=False)

    if debug:
        print("[DEBUG] sample_id:", sample.get("id", "unknown"))
        print("[DEBUG] prompt_len:", prompt_len)
        print("[DEBUG] out_len:", out_len)
        print("[DEBUG] raw_decode:")
        print(raw_decode)
        print("-" * 80)

    # When generate() is driven by inputs_embeds, some models may return only
    # newly generated tokens (or shorter sequence than prompt_len).
    if out_len > prompt_len:
        gen_ids = out_ids[0][prompt_len:]
        text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    else:
        text = tokenizer.decode(out_ids[0], skip_special_tokens=True).strip()
    return text


def _write_results(rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    payload = {
        "timestamp": ts,
        "model": "OpenGVLab/InternVL2-8B",
        "model_key": MODEL_NAME,
        "checkpoint": args.scale_path,
        "max_new_tokens": args.max_new_tokens,
        "samples": rows,
    }
    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    lines: list[str] = [
        "# InternVL2 Prompt Compare",
        "",
        f"- Time: {ts}",
        "- Model: `OpenGVLab/InternVL2-8B`",
        f"- Checkpoint: `{args.scale_path}`",
        f"- Max new tokens: `{args.max_new_tokens}`",
        "",
    ]
    for i, r in enumerate(rows):
        lines.extend(
            [
                f"## Sample {i + 1} ({r['id']})",
                "",
                "### Prompt",
                r["prompt"],
                "",
            ]
        )
        if r.get("image"):
            lines.extend([f"Image: `{r['image']}`", ""])
        lines.extend(["### FP16", r["fp16"], ""])
        if not args.fp16_only:
            lines.extend(["### Quantized checkpoint", r["quant"], ""])
        lines.extend(["---", ""])

    os.makedirs(os.path.dirname(args.output_md) or ".", exist_ok=True)
    with open(args.output_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    args = parse_args()
    if not args.fp16_only and not args.scale_path:
        raise ValueError("Please provide --scale_path, or set --fp16_only.")

    samples = _build_samples(args)
    rows: list[dict[str, Any]] = []

    lm_fp16, pm_fp16 = _load_internvl2(args.batch_size, args.device, scale_path=None)
    for s in samples:
        fp16_text = _generate_text(pm_fp16, s, args.max_new_tokens, debug=args.debug)
        rows.append(
            {
                "id": s.get("id", "sample"),
                "prompt": s["prompt"],
                "image": s.get("image"),
                "fp16": fp16_text,
                "quant": "",
            }
        )
    del pm_fp16, lm_fp16
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if not args.fp16_only:
        lm_quant, pm_quant = _load_internvl2(args.batch_size, args.device, args.scale_path)
        for i, s in enumerate(samples):
            rows[i]["quant"] = _generate_text(pm_quant, s, args.max_new_tokens, debug=args.debug)
        del pm_quant, lm_quant
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    _write_results(rows, args)
    for r in rows:
        print("=" * 80)
        print(f"[{r['id']}]")
        print(f"Prompt: {r['prompt']}")
        if r.get("image"):
            print(f"Image: {r['image']}")
        print(f"FP16 : {r['fp16']}")
        if not args.fp16_only:
            print(f"Quant: {r['quant']}")
    print("=" * 80)
    print(f"Saved JSON: {args.output_json}")
    print(f"Saved MD  : {args.output_md}")


if __name__ == "__main__":
    main()
