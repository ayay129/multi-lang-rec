import os
from typing import Dict

import yaml
from paddleocr import TextDetection, TextRecognition

default_confs_dir = os.path.join(os.path.dirname(__file__), "default_confs")
config_dir = os.path.join(default_confs_dir, "config.yaml")


def _build_model_kwargs(model_name: str, model_dir: str) -> dict:
    kwargs = {
        "model_name": model_name,
        "model_dir": model_dir,
    }
    device = os.getenv("OCR_DEVICE")
    if device:
        kwargs["device"] = device
    return kwargs


def init_models() -> Dict[str, object]:
    with open(config_dir, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    model_root = config["model_dir"]
    det_name = config["detection"]["model_name"]
    rec_config = config["recognition"]

    detector = TextDetection(
        **_build_model_kwargs(det_name, f"{model_root}/{det_name}")
    )

    recognizers = {}
    for lang, rec in rec_config.items():
        rec_model_name = rec["model_name"]
        recognizers[lang] = TextRecognition(
            **_build_model_kwargs(rec_model_name, f"{model_root}/{rec_model_name}")
        )

    return {
        "detector": detector,
        "recognizers": recognizers,
    }


ocr_components = init_models()

