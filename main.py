"""
ASDQ entry: load model + COCO calibration data.
Quantization pipeline not implemented yet; this step only verifies the basic architecture.
"""
import argparse
import os
import sys
import warnings
from typing import Union

import yaml

warnings.simplefilter("ignore", category=DeprecationWarning)

from lmms_eval.models import get_model

from asdq.models import get_process_model
from asdq.calibration.coco_vl import get_multimodal_calib_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="ASDQ: load model and COCO calibration data (no quantization yet).",
    )
    parser.add_argument(
        "--config",
        default="",
        help="Path to a yaml file; if set, CLI args are overridden by config.",
    )
    parser.add_argument("--model", default="llava_onevision", help="Model name (e.g. llava_onevision)")
    parser.add_argument(
        "--model_args",
        default="",
        help="Model arguments string, e.g. pretrained=path/to/model,dtype=float16",
    )
    parser.add_argument(
        "--batch_size", "-b", type=str, default="1", metavar="auto|auto:N|N", help="Batch size. Default 1.",
    )
    parser.add_argument("--device", type=str, default=None, help="Device (e.g. cuda, cuda:0, cpu)")
    parser.add_argument("--calib_data", default="coco", choices=["coco"], help="Calibration data source (only coco for now)")
    parser.add_argument("--n_samples", default=128, type=int, help="Number of calibration samples")
    parser.add_argument("--data_path", default="", type=str, help="Path to COCO JSON/JSONL")
    parser.add_argument("--image_folder", default="", type=str, help="Path to image directory for COCO")
    parser.add_argument("--interleave_format", action="store_true")
    parser.add_argument("--few_shot_format", action="store_true")
    parser.add_argument("--text_data_path", default="", type=str)
    args = parser.parse_args()
    return args


def cli_main(args: Union[argparse.Namespace, None] = None) -> None:
    if args is None:
        args = parse_args()

    if args.config:
        if not os.path.exists(args.config):
            raise FileNotFoundError(f"Config file does not exist: {args.config}")
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
        config = [config] if not isinstance(config, list) else config
        for cfg in config:
            args_copy = argparse.Namespace(**vars(args))
            for k, v in cfg.items():
                setattr(args_copy, k, v)
            _run_single(args_copy)
    else:
        _run_single(args)


def _run_single(args: argparse.Namespace) -> None:
    if args.model_args is None:
        args.model_args = ""

    # 1) Load model via lmms-eval
    ModelClass = get_model(args.model)
    lm = ModelClass.create_from_arg_string(
        args.model_args,
        {"batch_size": args.batch_size, "device": args.device},
    )

    # 2) Wrap with ASDQ process model (for calibration interface)
    ProcessModelClass = get_process_model(args.model)
    process_model = ProcessModelClass(
        lm._model,
        lm._tokenizer,
        lm.processor if hasattr(lm, "processor") else None,
    )

    # 3) Load COCO calibration data
    prompt_inputs = None
    prompt_kwargs = None

    if args.calib_data == "coco":
        if not args.data_path or not args.image_folder:
            print("Warning: --data_path and --image_folder are required for calib_data=coco. Skipping calibration load.")
        else:
            prompt_inputs, prompt_kwargs = get_multimodal_calib_dataset(
                data_path=args.data_path,
                image_folder=args.image_folder,
                model=process_model,
                n_samples=args.n_samples,
                few_shot_format=args.few_shot_format,
                interleave_format=args.interleave_format,
                text_data_path=args.text_data_path or None,
            )
            print("Calibration data loaded: prompt_inputs and prompt_kwargs ready.")

    # 4) Placeholder: no quantization yet
    print("[ASDQ] Model and calibration load completed. Quantization pipeline not implemented yet.")


if __name__ == "__main__":
    cli_main()
