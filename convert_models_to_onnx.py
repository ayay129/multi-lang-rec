#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert all Paddle models under `models/` to ONNX using paddle2onnx.

Usage:
  python convert_models_to_onnx.py [--models_dir /abs/path/to/models] [--force]

- By default, scans `./models` relative to this script.
- Creates `onnx/model.onnx` inside each model directory.
- Determines model/params filenames automatically (prefers `.json` as per user's example).
"""
import os
import sys
import subprocess
import logging
from typing import Optional, Tuple, List
from pathlib import Path

# ---- Logging setup ----
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("convert2onnx")

# ---- Helpers ----
MODEL_CANDIDATES: List[str] = [
    "inference.json",
    "model.json",
    "inference.pdmodel",
    "model.pdmodel",
]
PARAMS_CANDIDATES: List[str] = [
    "inference.pdiparams",
    "model.pdiparams",
]


def find_model_files(model_dir: Path) -> Optional[Tuple[str, str]]:
    """Try to find (model_filename, params_filename) under a model_dir.
    Returns None if not found.
    """
    model_filename = None
    params_filename = None

    # Pick first existing file from candidates
    for name in MODEL_CANDIDATES:
        if (model_dir / name).exists():
            model_filename = name
            break
    for name in PARAMS_CANDIDATES:
        if (model_dir / name).exists():
            params_filename = name
            break

    if model_filename and params_filename:
        return model_filename, params_filename
    return None


def ensure_paddle2onnx_available() -> bool:
    """Check if paddle2onnx CLI is available."""
    try:
        subprocess.run(["paddle2onnx", "--help"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        return True
    except FileNotFoundError:
        return False


def convert_model(model_dir: Path, save_path: Path, model_filename: str, params_filename: str) -> bool:
    """Run paddle2onnx conversion and return success status."""
    cmd = [
        "paddle2onnx",
        "--model_dir", str(model_dir),
        "--model_filename", model_filename,
        "--params_filename", params_filename,
        "--save_file", str(save_path),
        # Optional flags you may want to tune:
        # "--opset_version", "10",
        # "--enable_onnx_checker", "True",
    ]
    logger.info(f"Executing: {' '.join(cmd)}")
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode == 0:
        logger.info(f"Success → {save_path}")
        return True
    else:
        logger.error(f"Failed converting {model_dir.name}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
        return False


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Convert Paddle models under models/ to ONNX using paddle2onnx")
    parser.add_argument("--models_dir", type=str, default=str(Path(__file__).resolve().parent / "models"), help="Path to models directory")
    parser.add_argument("--force", action="store_true", help="Overwrite existing ONNX models if present")
    args = parser.parse_args()

    models_dir = Path(args.models_dir).resolve()
    if not models_dir.exists() or not models_dir.is_dir():
        logger.error(f"Models directory not found: {models_dir}")
        sys.exit(1)

    if not ensure_paddle2onnx_available():
        logger.error("paddle2onnx not found. Install with: pip install paddle2onnx")
        sys.exit(1)

    # Iterate subdirectories
    count_total = 0
    count_converted = 0
    count_skipped = 0
    count_failed = 0

    for child in sorted(models_dir.iterdir()):
        if not child.is_dir():
            continue
        # skip some directories that are not paddle models
        if child.name.lower() in {"torch"}:
            logger.info(f"Skip non-Paddle dir: {child.name}")
            continue

        # Check files
        found = find_model_files(child)
        if not found:
            logger.info(f"No model files found in {child.name} (expected one of {MODEL_CANDIDATES} and {PARAMS_CANDIDATES})")
            continue

        model_filename, params_filename = found
        count_total += 1

        # Prepare output path: <model_dir>/onnx/model.onnx
        onnx_dir = child / "onnx"
        onnx_dir.mkdir(exist_ok=True)
        save_path = onnx_dir / "model.onnx"

        if save_path.exists() and not args.force:
            logger.info(f"ONNX already exists, skip: {save_path} (use --force to overwrite)")
            count_skipped += 1
            continue
        import pdb;pdb.set_trace()

        ok = convert_model(child, save_path, model_filename, params_filename)
        if ok:
            count_converted += 1
        else:
            count_failed += 1

    logger.info("=" * 60)
    logger.info(f"Scan dir: {models_dir}")
    logger.info(f"Total candidates: {count_total}")
    logger.info(f"Converted: {count_converted}")
    logger.info(f"Skipped: {count_skipped}")
    logger.info(f"Failed: {count_failed}")

    # Exit code semantics
    if count_failed > 0:
        sys.exit(2)
    sys.exit(0)


if __name__ == "__main__":
    main()