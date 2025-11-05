import asyncio
import os
from typing import List, Optional
import tempfile
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
from PIL import Image

from ocr_engine import ocr_engines





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

async def process_image_async(image_path: str, language: str):
    """异步处理单张图片"""
    return language,ocr_engines[language].predict(image_path)

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
    
    best_words = []
    best_scores = []
    best_languages = []
    best_bboxes = []
    
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
                candidates.append({
                    "word": words[word_idx],
                    "score": scores[word_idx],
                    "language": language,
                    "bbox": bboxes[word_idx] if word_idx < len(bboxes) else None
                })
        
        # 从候选词中选出置信度最高的
        if candidates:
            best_candidate = max(candidates, key=lambda x: x["score"])
            best_words.append(best_candidate["word"])
            best_scores.append(best_candidate["score"])
            best_languages.append(best_candidate["language"])
            best_bboxes.append(best_candidate["bbox"])
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
    
    # 验证文件类型
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="文件必须是图片格式")
    
    try:
        # 读取图片文件
        image_data = await file.read()
        file_name = file.filename
        file_path = os.path.join(tempfile.gettempdir(), file_name)
        with open(file_path, 'wb') as f:
            f.write(image_data)
        
        # 转换为numpy数组
        # 解析语言参数
        target_languages = None
        if languages:
            target_languages = [lang.strip() for lang in languages.split(',')]
        else:
            target_languages = list(ocr_engines.keys())
        tasks = []        
        for language in target_languages:
            tasks.append(process_image_async(file_path, language))
        
        # 等待所有任务完成
        results = await asyncio.gather(*tasks)
        all_data = []
        for lang, res in results:
            obj = {}
            obj["language"] = lang
            obj["bbox"] =  res[0]["rec_boxes"].tolist() 
            obj["words_result"]= res[0]["rec_texts"]
            obj["scores"] = res[0]["rec_scores"]
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
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"图片处理失败: {str(e)}")
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)



if __name__ == "__main__":
    print("启动多语言OCR FastAPI服务...")
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=80,
        reload=False,
        log_level="info",
        workers=1
    )
