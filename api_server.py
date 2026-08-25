import asyncio
import cv2
import io
import os
import re
import time
import traceback
import unicodedata
from typing import List, Optional
import tempfile
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
from PIL import Image
import logging
import sys
from datetime import datetime
import uuid

from ocr_engine import ocr_components

# 配置日志 - 确保在uvicorn启动前配置
def setup_logging():
    # 创建自定义格式器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 获取根日志记录器
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # 清除现有的处理器
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # 添加控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # 添加文件处理器
    file_handler = logging.FileHandler('ocr_api.log', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    # 减少watchfiles的日志级别，避免过多的文件变化检测日志
    logging.getLogger("watchfiles").setLevel(logging.WARNING)
    
    return logging.getLogger(__name__)

# 初始化日志
logger = setup_logging()


class OCRResponse(BaseModel):
    """OCR响应模型"""
    success: bool
    message: str
    results: List[str] = []
    total_processing_time: float
    languages_processed: List[str]

app = FastAPI(
    title="多语言OCR API",
    description="基于PaddleOCR的异步并行多语言OCR识别服务",
    version="1.0.0"
)

LANGUAGE_SCRIPT_MAP = {
    "ar": "arabic",
    "en": "latin",
    "fr": "latin",
    "ru": "cyrillic",
    "zh": "cjk",
}

async def process_image_async(text_crops: list[np.ndarray], language: str):
    """异步处理共享检测后的文本块"""
    try:
        start_time = time.time()
        if not text_crops:
            return language, [], []
        result = list(
            ocr_components["recognizers"][language].predict(
                text_crops,
                batch_size=max(
                    1,
                    min(ocr_components["rec_batch_size"], len(text_crops)),
                ),
            )
        )
        end_time = time.time()
        processing_time = end_time - start_time

        start_time_formatted = datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        end_time_formatted = datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

        logger.info(f"{language}语言模型识别耗时: {processing_time:.4f}秒. star: {start_time_formatted}, end: {end_time_formatted}")
        texts = [_to_json_safe(item["rec_text"]) for item in result]
        scores = [float(_to_json_safe(item["rec_score"])) for item in result]
        score_thresh = ocr_components["rec_score_thresh"]
        texts = [text if score >= score_thresh else "" for text, score in zip(texts, scores)]
        return language, texts, scores
    except Exception as e:
        logger.error(f"{language}语言模型识别失败，错误: {str(e)}")
        raise


def _validate_image_bytes(image_data: bytes) -> None:
    try:
        with Image.open(io.BytesIO(image_data)) as img:
            img.verify()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="文件必须是图片格式") from exc


def _order_quad_points(points: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype=np.float32)
    point_sums = points.sum(axis=1)
    point_diffs = np.diff(points, axis=1)
    rect[0] = points[np.argmin(point_sums)]
    rect[2] = points[np.argmax(point_sums)]
    rect[1] = points[np.argmin(point_diffs)]
    rect[3] = points[np.argmax(point_diffs)]
    return rect


def _crop_axis_aligned(image: np.ndarray, polygon: list) -> np.ndarray:
    left, top, right, bottom = _bbox_bounds(polygon)
    left = max(int(left), 0)
    top = max(int(top), 0)
    right = min(int(right), image.shape[1])
    bottom = min(int(bottom), image.shape[0])
    if right <= left or bottom <= top:
        return image[:1, :1].copy()
    return image[top:bottom, left:right].copy()


def _crop_text_region(image: np.ndarray, polygon: list) -> np.ndarray:
    points = np.array(polygon, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 2:
        return _crop_axis_aligned(image, polygon)
    if points.shape[0] != 4:
        return _crop_axis_aligned(image, polygon)

    rect = _order_quad_points(points)
    width_top = np.linalg.norm(rect[1] - rect[0])
    width_bottom = np.linalg.norm(rect[2] - rect[3])
    height_right = np.linalg.norm(rect[2] - rect[1])
    height_left = np.linalg.norm(rect[3] - rect[0])

    crop_width = max(int(round(max(width_top, width_bottom))), 1)
    crop_height = max(int(round(max(height_right, height_left))), 1)

    dst = np.array(
        [
            [0, 0],
            [crop_width - 1, 0],
            [crop_width - 1, crop_height - 1],
            [0, crop_height - 1],
        ],
        dtype=np.float32,
    )
    transform = cv2.getPerspectiveTransform(rect, dst)
    crop = cv2.warpPerspective(
        image,
        transform,
        (crop_width, crop_height),
        borderMode=cv2.BORDER_REPLICATE,
        flags=cv2.INTER_CUBIC,
    )
    if crop_height / max(crop_width, 1) >= 1.5:
        crop = np.rot90(crop)
    return crop.copy()


def _detect_text_blocks(image: np.ndarray) -> list:
    start_time = time.time()
    result = list(ocr_components["detector"].predict(image, batch_size=1))
    processing_time = time.time() - start_time
    logger.info(f"共享检测模型耗时: {processing_time:.4f}秒")
    result_item = result[0]
    return _extract_result_boxes(result_item)


def _prepare_text_crops(image_path: str) -> tuple[list, list[np.ndarray]]:
    image = cv2.imread(image_path)
    if image is None:
        raise RuntimeError(f"无法读取图片: {image_path}")

    doc_preprocessor = ocr_components["doc_preprocessor"]
    if doc_preprocessor is not None:
        start_time = time.time()
        image = next(iter(doc_preprocessor.predict(image)))["output_img"]
        logger.info("共享文档预处理耗时: %.4f秒", time.time() - start_time)

    dt_polys = _detect_text_blocks(image)
    text_polys = []
    text_crops = []

    for polygon in dt_polys:
        normalized_polygon = _to_json_safe(polygon)
        crop = _crop_text_region(image, normalized_polygon)
        text_polys.append(normalized_polygon)
        text_crops.append(crop)

    textline_orientation = ocr_components["textline_orientation"]
    if textline_orientation is not None and text_crops:
        start_time = time.time()
        orientation_results = textline_orientation.predict(
            text_crops,
            batch_size=max(
                1,
                min(ocr_components["textline_batch_size"], len(text_crops)),
            ),
        )
        for index, result in enumerate(orientation_results):
            labels = _to_json_safe(result["label_names"])
            if labels and str(labels[0]).startswith("180"):
                text_crops[index] = np.rot90(text_crops[index], 2).copy()
        logger.info("共享文本行方向识别耗时: %.4f秒", time.time() - start_time)

    return text_polys, text_crops


def _detect_char_script(char: str) -> Optional[str]:
    code = ord(char)
    if 0x0400 <= code <= 0x052F or 0x2DE0 <= code <= 0x2DFF or 0xA640 <= code <= 0xA69F:
        return "cyrillic"
    if 0x0600 <= code <= 0x06FF or 0x0750 <= code <= 0x077F or 0x08A0 <= code <= 0x08FF:
        return "arabic"
    if 0x4E00 <= code <= 0x9FFF or 0x3400 <= code <= 0x4DBF:
        return "cjk"
    if char.isalpha():
        try:
            if "LATIN" in unicodedata.name(char):
                return "latin"
        except ValueError:
            return None
    return None


def _analyze_text(text: str) -> dict:
    compact_text = re.sub(r"\s+", "", text or "")
    letter_count = 0
    digit_count = 0
    punctuation_count = 0
    alnum_count = 0
    script_counts = {}

    for char in compact_text:
        if char.isalpha():
            letter_count += 1
            alnum_count += 1
            script = _detect_char_script(char)
            if script:
                script_counts[script] = script_counts.get(script, 0) + 1
        elif char.isdigit():
            digit_count += 1
            alnum_count += 1
        else:
            punctuation_count += 1

    dominant_script = None
    dominant_script_count = 0
    if script_counts:
        dominant_script, dominant_script_count = max(script_counts.items(), key=lambda item: item[1])

    dominant_script_ratio = dominant_script_count / letter_count if letter_count else 0.0
    punctuation_ratio = punctuation_count / max(len(compact_text), 1)
    effective_len = alnum_count

    return {
        "compact_text": compact_text,
        "alnum_count": alnum_count,
        "letter_count": letter_count,
        "digit_count": digit_count,
        "effective_len": effective_len,
        "punctuation_count": punctuation_count,
        "punctuation_ratio": punctuation_ratio,
        "script_counts": script_counts,
        "dominant_script": dominant_script,
        "dominant_script_ratio": dominant_script_ratio,
    }


def _bbox_dimensions(bbox) -> tuple[float, float]:
    if bbox is None:
        return 0.0, 0.0

    try:
        if len(bbox) == 0:
            return 0.0, 0.0
    except TypeError:
        return 0.0, 0.0

    if (
        isinstance(bbox, (list, tuple))
        and len(bbox) == 4
        and all(isinstance(value, (int, float)) for value in bbox)
    ):
        x1, y1, x2, y2 = bbox
        return float(abs(x2 - x1)), float(abs(y2 - y1))

    if isinstance(bbox, (list, tuple)):
        points = [
            point for point in bbox
            if isinstance(point, (list, tuple)) and len(point) >= 2
        ]
        if points:
            xs = [point[0] for point in points]
            ys = [point[1] for point in points]
            return float(max(xs) - min(xs)), float(max(ys) - min(ys))

    return 0.0, 0.0


def _bbox_bounds(bbox) -> tuple[float, float, float, float]:
    if bbox is None:
        return 0.0, 0.0, 0.0, 0.0

    try:
        if len(bbox) == 0:
            return 0.0, 0.0, 0.0, 0.0
    except TypeError:
        return 0.0, 0.0, 0.0, 0.0

    if (
        isinstance(bbox, (list, tuple))
        and len(bbox) == 4
        and all(isinstance(value, (int, float)) for value in bbox)
    ):
        x1, y1, x2, y2 = bbox
        left, right = sorted((float(x1), float(x2)))
        top, bottom = sorted((float(y1), float(y2)))
        return left, top, right, bottom

    if isinstance(bbox, (list, tuple)):
        points = [
            point for point in bbox
            if isinstance(point, (list, tuple)) and len(point) >= 2
        ]
        if points:
            xs = [float(point[0]) for point in points]
            ys = [float(point[1]) for point in points]
            return min(xs), min(ys), max(xs), max(ys)

    return 0.0, 0.0, 0.0, 0.0


def sort_blocks_in_reading_order(blocks: list[dict]) -> tuple[list[dict], list[int]]:
    if not blocks:
        return [], []

    enriched_blocks = []
    for index, block in enumerate(blocks):
        left, top, right, bottom = _bbox_bounds(block.get("bbox"))
        height = max(bottom - top, 1.0)
        enriched_blocks.append({
            **block,
            "_raw_index": index,
            "_left": left,
            "_top": top,
            "_right": right,
            "_bottom": bottom,
            "_height": height,
            "_center_y": top + height / 2.0,
        })

    enriched_blocks.sort(key=lambda item: (item["_top"], item["_left"]))

    rows = []
    for block in enriched_blocks:
        placed = False
        for row in rows:
            row_height = max(row["bottom"] - row["top"], 1.0)
            center_threshold = max(block["_height"], row_height) * 0.6
            overlap = min(row["bottom"], block["_bottom"]) - max(row["top"], block["_top"])
            if (
                abs(block["_center_y"] - row["center_y"]) <= center_threshold
                or overlap >= min(block["_height"], row_height) * 0.3
            ):
                row["items"].append(block)
                row["top"] = min(row["top"], block["_top"])
                row["bottom"] = max(row["bottom"], block["_bottom"])
                row["center_y"] = sum(item["_center_y"] for item in row["items"]) / len(row["items"])
                placed = True
                break
        if not placed:
            rows.append({
                "items": [block],
                "top": block["_top"],
                "bottom": block["_bottom"],
                "center_y": block["_center_y"],
            })

    rows.sort(key=lambda row: row["top"])
    ordered_blocks = []
    reading_order_indices = []

    for row in rows:
        row["items"].sort(key=lambda item: (item["_left"], item["_top"]))
        for item in row["items"]:
            ordered_blocks.append({
                key: value
                for key, value in item.items()
                if not key.startswith("_")
            })
            reading_order_indices.append(item["_raw_index"])

    return ordered_blocks, reading_order_indices


def _extract_result_boxes(result_item) -> list:
    if "dt_polys" in result_item and result_item["dt_polys"] is not None:
        dt_polys = result_item["dt_polys"]
        return _to_json_safe(dt_polys.tolist() if hasattr(dt_polys, "tolist") else list(dt_polys))

    if "rec_boxes" in result_item and result_item["rec_boxes"] is not None:
        rec_boxes = result_item["rec_boxes"]
        return _to_json_safe(rec_boxes.tolist() if hasattr(rec_boxes, "tolist") else list(rec_boxes))

    return []


def _to_json_safe(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {key: _to_json_safe(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_to_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_to_json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def _get_group_dominant_script(candidates: list) -> Optional[str]:
    aggregated = {}
    for candidate in candidates:
        if candidate["analysis"]["letter_count"] < 2:
            continue
        for script, count in candidate["analysis"]["script_counts"].items():
            aggregated[script] = aggregated.get(script, 0) + count
    if not aggregated:
        return None
    dominant_script, dominant_count = max(aggregated.items(), key=lambda item: item[1])
    total_letters = sum(aggregated.values())
    if total_letters == 0 or dominant_count / total_letters < 0.45:
        return None
    return dominant_script


def _get_trusted_non_latin_script(candidates: list) -> Optional[str]:
    trusted_script = None
    trusted_weight = 0.0

    for candidate in candidates:
        analysis = candidate["analysis"]
        candidate_script = analysis["dominant_script"]
        language_script = LANGUAGE_SCRIPT_MAP.get(candidate["language"])

        if candidate_script not in {"cyrillic", "arabic", "cjk"}:
            continue
        if language_script != candidate_script:
            continue
        if analysis["letter_count"] < 6 or analysis["dominant_script_ratio"] < 0.55:
            continue

        weight = float(candidate["score"]) * analysis["effective_len"]
        if weight > trusted_weight:
            trusted_script = candidate_script
            trusted_weight = weight

    return trusted_script


def _compute_adjusted_score(
    candidate: dict,
    candidates: list,
    group_dominant_script: Optional[str],
    trusted_non_latin_script: Optional[str],
) -> float:
    analysis = candidate["analysis"]
    score = float(candidate["score"])
    effective_len = analysis["effective_len"]
    punctuation_ratio = analysis["punctuation_ratio"]
    bbox_width, bbox_height = _bbox_dimensions(candidate["bbox"])
    aspect_ratio = bbox_width / max(bbox_height, 1.0)
    max_peer_len = max(item["analysis"]["effective_len"] for item in candidates)
    max_peer_density = max(
        item["analysis"]["effective_len"] / max(_bbox_dimensions(item["bbox"])[0], 1.0)
        for item in candidates
    )
    relative_len_ratio = effective_len / max(max_peer_len, 1)
    density = effective_len / max(bbox_width, 1.0)

    length_factor = 1.0
    if effective_len == 0:
        length_factor *= 0.03
    elif max_peer_len >= 8 and effective_len <= 2:
        length_factor *= 0.08
    elif max_peer_len >= 8 and relative_len_ratio < 0.25:
        length_factor *= 0.35
    elif max_peer_len >= 5 and relative_len_ratio < 0.4:
        length_factor *= 0.72

    info_factor = 1.0 - min(punctuation_ratio, 0.85) * 0.8
    if analysis["letter_count"] == 0 and analysis["digit_count"] == 0:
        info_factor *= 0.2

    coverage_factor = 1.0
    if bbox_width >= 500 and aspect_ratio >= 5 and effective_len <= 2 and max_peer_len >= 8:
        coverage_factor *= 0.08
    elif bbox_width >= 500 and aspect_ratio >= 5 and relative_len_ratio < 0.3:
        coverage_factor *= 0.4
    elif bbox_width >= 300 and aspect_ratio >= 4 and effective_len <= 2 and max_peer_len >= 6:
        coverage_factor *= 0.15

    if max_peer_density > 0 and bbox_width >= 300:
        density_ratio = density / max_peer_density
        if max_peer_len >= 8 and density_ratio < 0.2:
            coverage_factor *= 0.25
        elif max_peer_len >= 5 and density_ratio < 0.35:
            coverage_factor *= 0.6

    if max_peer_len >= 8 and effective_len <= 2 and punctuation_ratio >= 0.5:
        coverage_factor *= 0.1

    script_factor = 1.0
    candidate_script = analysis["dominant_script"]
    if group_dominant_script and analysis["letter_count"] >= 6:
        if candidate_script == group_dominant_script and analysis["dominant_script_ratio"] >= 0.5:
            script_factor *= 1.08
        elif candidate_script and candidate_script != group_dominant_script:
            script_factor *= 0.92

    language_script = LANGUAGE_SCRIPT_MAP.get(candidate["language"])
    if language_script and analysis["letter_count"] >= 6:
        if candidate_script == language_script and analysis["dominant_script_ratio"] >= 0.5:
            script_factor *= 1.05
        elif candidate_script and candidate_script != language_script:
            script_factor *= 0.95

    if trusted_non_latin_script and relative_len_ratio >= 0.6 and analysis["letter_count"] >= 6:
        if candidate_script == trusted_non_latin_script and analysis["dominant_script_ratio"] >= 0.5:
            script_factor *= 1.18
        elif candidate_script and candidate_script != trusted_non_latin_script:
            script_factor *= 0.88

    return score * length_factor * info_factor * coverage_factor * script_factor


def _select_best_candidate(candidates: list) -> dict:
    group_dominant_script = _get_group_dominant_script(candidates)
    trusted_non_latin_script = _get_trusted_non_latin_script(candidates)
    best_candidate = None
    best_adjusted_score = -1.0

    for candidate in candidates:
        candidate["adjusted_score"] = _compute_adjusted_score(
            candidate,
            candidates,
            group_dominant_script,
            trusted_non_latin_script,
        )
        if candidate["adjusted_score"] > best_adjusted_score:
            best_candidate = candidate
            best_adjusted_score = candidate["adjusted_score"]

    if best_candidate:
        logger.info(
            "融合选择: word=%r lang=%s raw=%.4f adjusted=%.4f dominant_script=%s",
            best_candidate["word"],
            best_candidate["language"],
            float(best_candidate["score"]),
            best_candidate["adjusted_score"],
            group_dominant_script,
        )
    return best_candidate

def find_best_words_from_groups(data_groups):
    """
    从多组OCR结果中，按词的位置对比，选出每个位置置信度最高的词
    
    Args:
        data_groups: 包含多组OCR结果的列表
        
    Returns:
        dict: 包含最佳结果的字典
    """
    if not data_groups:
        return {"error": "No data groups provided"}
    
    # 找出所有组中词数量的最大值
    max_words = max(len(group.get("words_result", [])) for group in data_groups)
    
    best_blocks = []
    
    # 对每个词的位置进行比较
    for word_idx in range(max_words):
        candidates = []  # 存储当前位置所有组的候选词
        
        for group in data_groups:
            words = group.get("words_result", [])
            scores = group.get("scores", [])
            language = group.get("language", "unknown")
            bboxes = group.get("bbox", [])
            
            # 如果当前组在这个位置有词
            if word_idx < len(words) and word_idx < len(scores):
                word = words[word_idx]
                candidates.append({
                    "word": word,
                    "score": scores[word_idx],
                    "language": language,
                    "bbox": bboxes[word_idx] if word_idx < len(bboxes) else None,
                    "analysis": _analyze_text(word),
                })
        
        if candidates:
            best_candidate = _select_best_candidate(candidates)
            best_blocks.append({
                "word": best_candidate["word"],
                "score": best_candidate["score"],
                "language": best_candidate["language"],
                "bbox": best_candidate["bbox"],
            })

    raw_best_words = [block["word"] for block in best_blocks]
    ordered_blocks, reading_order_indices = sort_blocks_in_reading_order(best_blocks)
    best_words = [block["word"] for block in ordered_blocks]
    best_scores = [block["score"] for block in ordered_blocks]
    best_languages = [block["language"] for block in ordered_blocks]
    best_bboxes = [block["bbox"] for block in ordered_blocks]

    words_result = {} 
    for index, word in enumerate(best_words):
        words_result[index] = word
    bboxes = {}
    for index, bbox in enumerate(best_bboxes):
        bboxes[index] = bbox
    return {
        "language": ",".join(best_languages),
        "bbox": bboxes,
        "words_result": words_result,
        "words_num": len(" ".join(best_words)),
        "best_scores": best_scores,
        "combined_text": " ".join(best_words),
        "combined_text_raw": " ".join(raw_best_words),
        "reading_order_indices": reading_order_indices,
        "average_confidence": sum(best_scores) / len(best_scores) if best_scores else 0
    }

@app.post("/", response_model=OCRResponse)
async def ocr_upload_image(
    file: UploadFile = File(...),
    languages: Optional[str] = None
):
    """
    上传图片进行OCR识别
    
    Args:
        file: 上传的图片文件
        languages: 指定识别语言，多个语言用逗号分隔，如 "en,ar,fr"。不指定则使用所有语言
    
    Returns:
        OCRResponse: OCR识别结果
    """
    
    try:
        # 读取图片文件
        image_data = await file.read()
        _validate_image_bytes(image_data)
        
        # 生成唯一的文件名，保留原始扩展名
        file_extension = os.path.splitext(file.filename)[1] if file.filename else '.jpg'
        unique_filename = f"{os.path.splitext(file.filename)[0]}_{uuid.uuid4().hex}{file_extension}"
        file_path = os.path.join(tempfile.gettempdir(), unique_filename)
        logger.info(f"临时文件路径: {file_path}")
        with open(file_path, 'wb') as f:
            f.write(image_data)
        
        # 转换为numpy数组
        target_languages = None
        if languages:
            target_languages = [lang.strip() for lang in languages.split(',')]
        else:
            target_languages = list(ocr_components["recognizers"].keys())
        shared_bboxes, text_crops = _prepare_text_crops(file_path)
        tasks = []        
        for language in target_languages:
            tasks.append(process_image_async(text_crops, language))
        
        logger.info(f"开始执行 {len(tasks)} 个异步OCR任务...")
        async_start_time = time.time()
        results = await asyncio.gather(*tasks)
        async_total_time = time.time() - async_start_time
        logger.info(f"所有异步OCR任务完成，总异步耗时: {async_total_time:.4f}秒")
        all_data = []
        for lang, texts, scores in results:
            obj = {}
            obj["language"] = lang
            obj["bbox"] = shared_bboxes
            obj["words_result"] = texts
            obj["scores"] = scores
            all_data.append(obj)
        best_data = find_best_words_from_groups(all_data)
        #兼容旧格式
        for obj in all_data:
            w_results = obj["words_result"]
            o_bboxes = obj["bbox"]
            obj["words_result"] = {}
            obj["bbox"] = {}
            for index, word in enumerate(w_results):
                obj["words_result"][index] = word
            for index, box in enumerate(o_bboxes):
                obj["bbox"][index] = box
        best_data["rank_reco_resp"] = all_data
        return JSONResponse(content=best_data)
        
    except Exception as e:
        logger.info(f"删除临时文件: {file_path}")
        if os.path.exists(file_path):
            os.remove(file_path)
        logger.error(f"图片处理失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"图片处理失败: {str(e)}")
    finally:
        logger.info(f"删除临时文件: {file_path}")
        if os.path.exists(file_path):
            os.remove(file_path)



if __name__ == "__main__":
    host,port,workers,log_level,reload = "0.0.0.0", 8000, 1, "info", True
    logger.info("=" * 50)
    logger.info("启动多语言OCR FastAPI服务...")
    logger.info(f"可用的OCR引擎: {list(ocr_components['recognizers'].keys())}")
    logger.info(f"Pipeline配置: {ocr_components['pipeline_config_path']}")
    logger.info("服务配置:")
    logger.info(f"  - 主机: {host}")
    logger.info(f"  - 端口: {port}")
    logger.info(f"  - 工作进程数: {workers}")
    logger.info(f"  - 重载模式: {'启用' if reload else '禁用'}")
    logger.info(f"  - 日志级别: {log_level}")
    logger.info("=" * 50)
    
    try:
        uvicorn.run(
            app,
            host=host,
            port=port,
            reload=reload,
            log_level=log_level,
            access_log=True,
            use_colors=True,
            workers=workers
        )
    except Exception as e:
        logger.error(f"服务启动失败: {str(e)}")
        logger.error(f"详细错误信息: {traceback.format_exc()}")
        raise
