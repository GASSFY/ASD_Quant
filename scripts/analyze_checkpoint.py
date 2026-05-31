#!/usr/bin/env python3
"""Print byte breakdown of an ASDQ scale_path checkpoint."""
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from asdq.quantization.checkpoint import summarize_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze ASDQ checkpoint size by component")
    parser.add_argument("checkpoint", type=str, help="Path to .pt checkpoint")
    args = parser.parse_args()

    summary = summarize_checkpoint(args.checkpoint)
    total_gb = summary["total_bytes"] / (1024 ** 3)
    print(f"path: {summary['path']}")
    print(f"format: {summary['format']}")
    print(f"total: {summary['total_bytes'] / (1024 ** 2):.1f} MiB ({total_gb:.2f} GiB)")
    print("groups:")
    for name, nbytes in sorted(summary["groups"].items(), key=lambda x: -x[1]):
        print(f"  {name:24s} {nbytes / (1024 ** 2):8.1f} MiB")


if __name__ == "__main__":
    main()
