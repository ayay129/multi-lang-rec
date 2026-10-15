import copy
import os
from typing import Dict

import yaml
from paddlex import create_pipeline, create_predictor


base_dir = os.path.dirname(__file__)
default_confs_dir = os.path.join(base_dir, "default_confs")
config_dir = os.path.join(default_confs_dir, "config.yaml")


def _resolve_model_dir(model_dir: str | None) -> str | None:
    if not model_dir or os.path.isabs(model_dir):
        return model_dir
    return os.path.join(base_dir, model_dir)


def _model_kwargs(model_name: str, model_dir: str | None) -> dict:
    kwargs = {
        "model_name": model_name,
        "model_dir": _resolve_model_dir(model_dir),
    }
    device = os.getenv("OCR_DEVICE")
    if device:
        kwargs["device"] = device
    return kwargs


def _pipeline_config_path() -> str:
    configured_path = os.getenv("OCR_PIPELINE_CONFIG")
    if configured_path:
        if os.path.isabs(configured_path):
            return configured_path
        return os.path.join(base_dir, configured_path)

    filename = (
        "DefaultAtlasOCR.yaml"
        if os.getenv("PADDLEX_USE_AIS_BENCH") == "1"
        else "DefaultOCR.yaml"
    )
    return os.path.join(default_confs_dir, filename)


def _load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def _create_doc_preprocessor(config: dict):
    if not config.get("use_doc_preprocessor", False):
        return None

    doc_config = copy.deepcopy(config["SubPipelines"]["DocPreprocessor"])
    for model_config in doc_config.get("SubModules", {}).values():
        model_config["model_dir"] = _resolve_model_dir(model_config.get("model_dir"))

    kwargs = {"config": doc_config}
    device = os.getenv("OCR_DEVICE")
    if device:
        kwargs["device"] = device
    return create_pipeline(**kwargs)


def init_models() -> Dict[str, object]:
    model_config = _load_yaml(config_dir)
    pipeline_config_path = _pipeline_config_path()
    pipeline_config = _load_yaml(pipeline_config_path)
    submodules = pipeline_config["SubModules"]

    model_root = _resolve_model_dir(model_config["model_dir"])
    det_name = model_config["detection"]["model_name"]
    det_config = submodules["TextDetection"]
    detector = create_predictor(
        **_model_kwargs(det_name, os.path.join(model_root, det_name)),
        **{
            key: det_config.get(key)
            for key in (
                "limit_side_len",
                "limit_type",
                "thresh",
                "box_thresh",
                "unclip_ratio",
            )
        },
    )

    textline_config = submodules.get("TextLineOrientation", {})
    textline_orientation = None
    if pipeline_config.get("use_textline_orientation", False):
        textline_orientation = create_predictor(
            **_model_kwargs(
                textline_config["model_name"],
                textline_config.get("model_dir"),
            )
        )

    recognizers = {}
    recognizers_by_model = {}
    for lang, rec in model_config["recognition"].items():
        rec_model_name = rec["model_name"]
        if rec_model_name not in recognizers_by_model:
            recognizers_by_model[rec_model_name] = create_predictor(
                **_model_kwargs(rec_model_name, os.path.join(model_root, rec_model_name))
            )
        recognizers[lang] = recognizers_by_model[rec_model_name]

    rec_config = submodules["TextRecognition"]
    return {
        "pipeline_config_path": pipeline_config_path,
        "doc_preprocessor": _create_doc_preprocessor(pipeline_config),
        "detector": detector,
        "textline_orientation": textline_orientation,
        "textline_batch_size": int(textline_config.get("batch_size", 1)),
        "recognizers": recognizers,
        "rec_batch_size": int(rec_config.get("batch_size", 1)),
        "rec_score_thresh": float(rec_config.get("score_thresh", 0.0)),
    }


ocr_components = init_models()
