#!/usr/bin/env python3
"""
OCRBench (or other lmms-eval task) inference benchmark: latency, throughput, optional GPU power log.

Usage examples (run from ASDQ repo root):

  # FP16 baseline
  python scripts/bench_inference.py \\
    --model internvl2 \\
    --model_args "pretrained=OpenGVLab/InternVL2-8B" \\
    --tasks ocrbench \\
    --limit 100 --warmup 5

  # Real int4 checkpoint (v2)
  python scripts/bench_inference.py \\
    --model internvl2 \\
    --model_args "pretrained=OpenGVLab/InternVL2-8B" \\
    --tasks ocrbench \\
    --scale_path eval_new_results/OpenGVLab_InternVL2-8B/scale_cache/asdq_w4.pt \\
    --real_quant \\
    --limit 100 --warmup 5 \\
    --output_json bench_results/quant.json

  # With GPU power sampling (Windows/Linux + nvidia-smi)
  python scripts/bench_inference.py --config configs/bench_fp16.yaml --power_log bench_results/power.csv

Power is sampled via ``nvidia-smi --query-gpu=power.draw`` in a background thread (default 200 ms).
Average W is computed over the measured inference window only (warmup excluded).
Model loading time is NOT included in latency stats.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import subprocess
import sys
import threading
import time
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import MethodType
from typing import Any, List, Optional, Tuple

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

warnings.simplefilter("ignore", category=DeprecationWarning)

from lmms_eval import evaluator
from lmms_eval.models import get_model
from lmms_eval.tasks import TaskManager

from asdq.quantization.eval_load import load_model_for_eval, resolve_eval_load_mode


def _parse_seed(seed_str: str) -> tuple:
    parts = seed_str.replace(" ", "").split(",")
    if len(parts) == 1:
        try:
            v = int(parts[0])
            return (v, v, v, v)
        except ValueError:
            return (0, 1234, 1234, 1234)
    out = []
    for p in parts[:4]:
        try:
            out.append(int(p))
        except ValueError:
            out.append(1234)
    while len(out) < 4:
        out.append(1234)
    return tuple(out)


def _percentile(values: List[float], pct: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    k = (len(ordered) - 1) * (pct / 100.0)
    f = int(k)
    c = min(f + 1, len(ordered) - 1)
    if f == c:
        return ordered[f]
    return ordered[f] + (ordered[c] - ordered[f]) * (k - f)


def _count_int4_modules(model: torch.nn.Module) -> int:
    from asdq.quantization.real_quant import Int4QuantLinear

    return sum(1 for m in model.modules() if isinstance(m, Int4QuantLinear))


@dataclass
class MeasureWindow:
    start_ts: Optional[float] = None
    end_ts: Optional[float] = None


def _instrument_generate_until(
    lm: Any,
    warmup: int,
    limit: int,
    latencies_sec: List[float],
    measure_window: MeasureWindow,
) -> None:
    """Wrap generate_until to record per-sample E2E time (cuda synchronized)."""
    original = lm.generate_until
    sample_idx = {"n": 0}

    def generate_until_timed(requests):
        results = []
        for reg in requests:
            sample_idx["n"] += 1
            if sample_idx["n"] == warmup + 1:
                measure_window.start_ts = time.perf_counter()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            chunk = original([reg])
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            if sample_idx["n"] > warmup:
                latencies_sec.append(dt)
            if sample_idx["n"] == warmup + limit:
                measure_window.end_ts = time.perf_counter()
            results.extend(chunk)
        return results

    lm.generate_until = MethodType(lambda _self, reqs: generate_until_timed(reqs), lm)


def _query_gpu_power_draw() -> Optional[float]:
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        if r.returncode != 0 or not r.stdout.strip():
            return None
        val = float(r.stdout.strip().splitlines()[0].split(",")[0].strip())
        return val if val >= 0 else None
    except (FileNotFoundError, ValueError, IndexError, subprocess.TimeoutExpired):
        return None


class PowerSampler:
    """Background power sampler using nvidia-smi --query-gpu=power.draw."""

    def __init__(self, log_path: str, interval_ms: int = 200) -> None:
        self.log_path = log_path
        self.interval_sec = interval_ms / 1000.0
        self.samples: List[Tuple[float, float]] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._t0 = 0.0

    def start(self, t0: float) -> bool:
        if self._thread is not None:
            return True
        probe = _query_gpu_power_draw()
        if probe is None:
            print("[bench] nvidia-smi power.draw unavailable; skip power logging.")
            return False
        os.makedirs(os.path.dirname(self.log_path) or ".", exist_ok=True)
        self._t0 = t0
        self.samples = [(t0, probe)]
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return True

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=10)
        self._thread = None
        self._write_log()

    def _run(self) -> None:
        while not self._stop.is_set():
            if self._stop.wait(self.interval_sec):
                break
            watts = _query_gpu_power_draw()
            if watts is not None:
                self.samples.append((time.perf_counter(), watts))

    def _write_log(self) -> None:
        with open(self.log_path, "w", encoding="utf-8") as f:
            f.write("elapsed_sec,power_w\n")
            for ts, watts in self.samples:
                f.write(f"{ts - self._t0:.6f},{watts:.3f}\n")

    def summarize(self, window: MeasureWindow) -> dict:
        if window.start_ts is None or window.end_ts is None:
            return {
                "power_avg_w": None,
                "power_samples": 0,
                "power_measure_window_sec": None,
            }
        in_window = [
            watts
            for ts, watts in self.samples
            if window.start_ts <= ts <= window.end_ts
        ]
        if not in_window:
            return {
                "power_avg_w": None,
                "power_samples": 0,
                "power_measure_window_sec": round(window.end_ts - window.start_ts, 3),
            }
        return {
            "power_avg_w": float(statistics.mean(in_window)),
            "power_samples": len(in_window),
            "power_measure_window_sec": round(window.end_ts - window.start_ts, 3),
        }


def _parse_dmon_power_log(path: str) -> Optional[float]:
    """Return mean GPU power (W) from legacy nvidia-smi dmon -s p output (pwr column)."""
    if not os.path.isfile(path):
        return None
    pwr_col: Optional[int] = None
    powers: List[float] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                header = line.lstrip("#").split()
                if "pwr" in header:
                    pwr_col = header.index("pwr")
                continue
            if pwr_col is None:
                continue
            parts = line.split()
            if len(parts) <= pwr_col:
                continue
            token = parts[pwr_col]
            if token == "-":
                continue
            try:
                val = float(token)
                if val >= 0:
                    powers.append(val)
            except ValueError:
                continue
    if not powers:
        return None
    return float(statistics.mean(powers))


def _parse_query_power_log(path: str, window: Optional[MeasureWindow] = None, t0: float = 0.0) -> Optional[float]:
    """Return mean power (W) from query-sampler CSV (elapsed_sec,power_w)."""
    if not os.path.isfile(path):
        return None
    powers: List[float] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("elapsed") or line.startswith("#"):
                continue
            parts = line.split(",")
            if len(parts) < 2:
                continue
            try:
                elapsed = float(parts[0])
                watts = float(parts[1])
            except ValueError:
                continue
            if watts < 0:
                continue
            ts = t0 + elapsed
            if window is not None and window.start_ts is not None and window.end_ts is not None:
                if ts < window.start_ts or ts > window.end_ts:
                    continue
            powers.append(watts)
    if not powers:
        return None
    return float(statistics.mean(powers))


def _warn_suspicious_power(power_avg_w: Optional[float], gpu_name: str) -> None:
    if power_avg_w is None:
        return
    high_end_markers = ("PRO", "RTX", "A100", "H100", "A6000", "A5000", "TESLA", "QUADRO")
    name_upper = gpu_name.upper()
    if power_avg_w < 80 and any(marker in name_upper for marker in high_end_markers):
        print(
            f"[bench] WARNING: GPU power ({power_avg_w:.1f} W) looks unusually low for "
            f"{gpu_name}. Check driver, power limit, or whether a temperature column was "
            "parsed by mistake (legacy dmon logs)."
        )


def _summarize_latencies(latencies_sec: List[float]) -> dict:
    if not latencies_sec:
        return {
            "count": 0,
            "mean_s": float("nan"),
            "median_s": float("nan"),
            "p95_s": float("nan"),
            "mean_ms": float("nan"),
            "throughput_samples_per_s": float("nan"),
        }
    mean_s = statistics.mean(latencies_sec)
    mean_ms = mean_s * 1000.0
    return {
        "count": len(latencies_sec),
        "mean_s": mean_s,
        "median_s": statistics.median(latencies_sec),
        "p95_s": _percentile(latencies_sec, 95),
        "min_s": min(latencies_sec),
        "max_s": max(latencies_sec),
        "mean_ms": mean_ms,
        "throughput_samples_per_s": 1.0 / mean_s if mean_s > 0 else float("nan"),
        "throughput_formula_1000_over_mean_ms": 1000.0 / mean_ms if mean_ms > 0 else float("nan"),
    }


def _print_gpu_env() -> dict:
    info: dict = {"cuda_available": torch.cuda.is_available()}
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        info["gpu_total_gib"] = round(props.total_memory / (1024 ** 3), 2)
        try:
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version,power.draw", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                check=False,
            )
            if r.stdout.strip():
                info["nvidia_smi"] = r.stdout.strip()
        except FileNotFoundError:
            pass
    print("[bench] Environment:", json.dumps(info, ensure_ascii=False))
    return info


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ASDQ inference benchmark (latency / throughput / optional power)")
    p.add_argument("--config", default="", help="Optional yaml config (merged with CLI)")
    p.add_argument("--model", default="internvl2")
    p.add_argument("--model_args", default="pretrained=OpenGVLab/InternVL2-8B")
    p.add_argument("--tasks", default="ocrbench")
    p.add_argument("--batch_size", "-b", default="1")
    p.add_argument("--device", default=None)
    p.add_argument("--limit", type=int, default=100, help="Measured samples (after warmup)")
    p.add_argument("--warmup", type=int, default=5, help="Warmup samples (excluded from stats)")
    p.add_argument("--scale_path", default=None)
    p.add_argument("--real_quant", action="store_true", default=False)
    p.add_argument("--pseudo_quant", action="store_true", default=False)
    p.add_argument("--seed", default="0,1234,1234,1234")
    p.add_argument("--verbosity", default="ERROR")
    p.add_argument(
        "--output_path",
        default="bench_results",
        help="lmms-eval task output dir (OCRBench writes submission files here)",
    )
    p.add_argument("--output_json", default="", help="Save summary JSON here")
    p.add_argument("--power_log", default="", help="If set, sample power.draw to this CSV during inference")
    p.add_argument("--power_interval_ms", type=int, default=200, help="Power sampling interval (ms)")
    p.add_argument("--run_tag", default="", help="Label in JSON, e.g. fp16 or int4")
    return p.parse_args()


def _apply_yaml(args: argparse.Namespace) -> None:
    if not args.config or not os.path.isfile(args.config):
        return
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    for k, v in cfg.items():
        if hasattr(args, k):
            setattr(args, k, v)


def run_benchmark(args: argparse.Namespace) -> dict:
    gpu_env = _print_gpu_env()
    gpu_name = gpu_env.get("gpu_name", "")
    load_mode = resolve_eval_load_mode(
        args.scale_path,
        real_quant=args.real_quant,
        pseudo_quant=args.pseudo_quant,
    )
    run_tag = args.run_tag or load_mode
    print(f"[bench] Run tag: {run_tag} | load mode: {load_mode}")
    print(f"[bench] Tasks: {args.tasks} | warmup: {args.warmup} | measure limit: {args.limit}")

    task_manager = TaskManager(args.verbosity, model_name=args.model)
    task_names = task_manager.match_tasks([t.strip() for t in args.tasks.split(",")])

    print("[bench] Loading model (not timed)...")
    t_load0 = time.perf_counter()
    ModelClass = get_model(args.model)
    lm = load_model_for_eval(ModelClass, args)
    load_sec = time.perf_counter() - t_load0
    int4_layers = _count_int4_modules(lm._model)
    print(f"[bench] Model load: {load_sec:.1f}s | Int4QuantLinear modules: {int4_layers}")

    latencies: List[float] = []
    measure_window = MeasureWindow()
    _instrument_generate_until(
        lm,
        warmup=args.warmup,
        limit=args.limit,
        latencies_sec=latencies,
        measure_window=measure_window,
    )

    seeds = _parse_seed(args.seed)
    random.seed(seeds[0])
    np.random.seed(seeds[1])
    torch.manual_seed(seeds[2])

    total_limit = args.warmup + args.limit
    os.makedirs(args.output_path, exist_ok=True)
    power_sampler: Optional[PowerSampler] = None

    print("[bench] Starting inference (timed)...")
    t_infer0 = time.perf_counter()
    if args.power_log:
        print(f"[bench] Power logging -> {args.power_log} (interval {args.power_interval_ms} ms)")
        power_sampler = PowerSampler(args.power_log, interval_ms=args.power_interval_ms)
        if not power_sampler.start(t_infer0):
            power_sampler = None

    try:
        evaluator.simple_evaluate(
            model=args.model,
            lm=lm,
            model_args=args.model_args,
            tasks=task_names,
            batch_size=args.batch_size,
            device=args.device,
            limit=total_limit,
            check_integrity=False,
            log_samples=False,
            task_manager=task_manager,
            verbosity=args.verbosity,
            random_seed=seeds[0],
            numpy_random_seed=seeds[1],
            torch_random_seed=seeds[2],
            fewshot_random_seed=seeds[3],
            cli_args=args,
        )
    finally:
        infer_sec = time.perf_counter() - t_infer0
        if power_sampler is not None:
            power_sampler.stop()

    stats = _summarize_latencies(latencies)
    power_stats = (
        power_sampler.summarize(measure_window)
        if power_sampler is not None
        else {
            "power_avg_w": None,
            "power_samples": 0,
            "power_measure_window_sec": None,
        }
    )
    power_avg_w = power_stats["power_avg_w"]
    _warn_suspicious_power(power_avg_w, gpu_name)

    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "run_tag": run_tag,
        "load_mode": load_mode,
        "model": args.model,
        "model_args": args.model_args,
        "scale_path": args.scale_path,
        "tasks": args.tasks,
        "warmup": args.warmup,
        "limit_measured": args.limit,
        "total_limit": total_limit,
        "int4_modules": int4_layers,
        "model_load_sec": round(load_sec, 3),
        "wall_infer_sec": round(infer_sec, 3),
        "latency": stats,
        "power_avg_w": power_avg_w,
        "power_samples": power_stats["power_samples"],
        "power_measure_window_sec": power_stats["power_measure_window_sec"],
        "power_interval_ms": args.power_interval_ms if args.power_log else None,
        "power_log": args.power_log or None,
        "notes": (
            "Per-sample E2E = one generate_until call (image prep + vision + LLM decode). "
            "Load time excluded. Power avg is over measured samples only (warmup excluded), "
            "sampled via nvidia-smi --query-gpu=power.draw."
        ),
    }

    print("\n[bench] ===== Results =====")
    print(f"  Samples measured : {stats['count']}")
    print(f"  Mean latency     : {stats['mean_s']:.4f} s/sample ({stats['mean_ms']:.1f} ms)")
    print(f"  Median latency   : {stats['median_s']:.4f} s")
    print(f"  P95 latency      : {stats['p95_s']:.4f} s")
    print(f"  Throughput       : {stats['throughput_samples_per_s']:.4f} samples/s")
    print(f"                    (= 1000 / {stats['mean_ms']:.1f} ms = {stats['throughput_formula_1000_over_mean_ms']:.4f})")
    if power_avg_w is not None:
        print(
            f"  GPU power (avg, measured window): {power_avg_w:.1f} W  "
            f"({power_stats['power_samples']} samples, "
            f"{power_stats['power_measure_window_sec']:.1f}s window, log: {args.power_log})"
        )
    else:
        print("  GPU power        : (not logged; use --power_log out.csv)")
    print("[bench] ====================\n")

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"[bench] Saved {args.output_json}")

    return summary


def main() -> None:
    args = parse_args()
    _apply_yaml(args)
    run_benchmark(args)


if __name__ == "__main__":
    main()
