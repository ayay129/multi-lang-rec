#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用 paddle2onnx 批量将 `models/` 目录下的 Paddle 模型转换为 ONNX。

用法：
  python convert_models_to_onnx.py [--models_dir 绝对路径到models目录] [--force]

说明：
- 默认扫描脚本所在目录下的 `./models`。
- 自动识别模型文件与参数文件（优先使用 `.json` 与 `.pdiparams`）。
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
        # 低于7 atc无法编译，默认11
        "--opset_version", "11"
        # "--enable_onnx_checker", "True",
    ]
    logger.info(f"执行命令: {' '.join(cmd)}")
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode == 0:
        logger.info(f"转换成功 → {save_path}")
        return True
    else:
        logger.error(f"转换失败: {model_dir.name}\n标准输出:\n{proc.stdout}\n标准错误:\n{proc.stderr}")
        return False


def main():
    import argparse
    parser = argparse.ArgumentParser(description="使用 paddle2onnx 将 models/ 下的 Paddle 模型转换为 ONNX")
    parser.add_argument("--models_dir", type=str, default=str(Path(__file__).resolve().parent / "models"), help="模型目录的路径")
    parser.add_argument("--force", action="store_true", help="若已存在 ONNX 文件则覆盖")
    args = parser.parse_args()

    models_dir = Path(args.models_dir).resolve()
    if not models_dir.exists() or not models_dir.is_dir():
        logger.error(f"未找到模型目录: {models_dir}")
        sys.exit(1)

    if not ensure_paddle2onnx_available():
        logger.error("未找到 paddle2onnx，请先安装: pip install paddle2onnx")
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
            logger.info(f"跳过非 Paddle 目录: {child.name}")
            continue

        # 检查模型文件
        found = find_model_files(child)
        if not found:
            logger.info(f"目录 {child.name} 中未找到模型文件（应存在 {MODEL_CANDIDATES} 与 {PARAMS_CANDIDATES} 之一）")
            continue

        model_filename, params_filename = found
        count_total += 1

        # 输出路径：<model_dir>/inference.onnx（不创建 onnx 子目录）
        save_path = child / "inference.onnx"

        if save_path.exists() and not args.force:
            logger.info(f"ONNX 已存在，跳过: {save_path}（使用 --force 可覆盖）")
            count_skipped += 1
            continue

        ok = convert_model(child, save_path, model_filename, params_filename)
        if ok:
            count_converted += 1
        else:
            count_failed += 1

    logger.info("=" * 60)
    logger.info(f"扫描目录: {models_dir}")
    logger.info(f"候选模型总数: {count_total}")
    logger.info(f"成功转换: {count_converted}")
    logger.info(f"已跳过: {count_skipped}")
    logger.info(f"失败数量: {count_failed}")

    # Exit code semantics
    if count_failed > 0:
        sys.exit(2)
    sys.exit(0)


if __name__ == "__main__":
    main()