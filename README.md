<<<<<<< README.md
# 多语言OCR FastAPI服务

**语言:** **中文** | [English](README_EN.md) | [العربية](README_AR.md)

基于PaddleOCR PPStructure产线逻辑设计的多语言精确识别架构，通过统一文本检测+多语言识别+置信度融合的方式，实现高精度的多语言文本识别服务。

## 架构设计

### 核心思想
本项目基于PaddleOCR的PPStructure产线逻辑，采用"统一检测+分语言识别+置信度融合"的创新架构：

1. **统一文本检测**: 使用单一的检测模型识别图像中所有文本区域（text blocks）
2. **多语言并行识别**: 将每个文本区域分别送入不同语言的识别模型进行识别
3. **置信度融合**: 对每个文本区域的多语言识别结果进行置信度比较，选择最优结果
4. **结果组合**: 将所有文本区域的最优识别结果组合成最终输出

### 技术优势
- 🎯 **精确定位**: 统一检测确保文本区域定位的一致性和准确性
- 🚀 **并行处理**: 多语言识别任务异步并行执行，大幅提升处理效率
- 🧠 **智能融合**: 基于置信度的结果选择，自动选择最佳识别结果
- 🌍 **语言无关**: 无需预先判断文本语言，系统自动选择最优识别结果

## 功能特性

- 🚀 **异步并行处理**: 同时对多种语言进行OCR识别，提高处理效率
- 🌍 **多语言支持**: 支持英语(en)、阿拉伯语(ar)、法语(fr)等多种语言
- 🎯 **精确识别**: 基于置信度的智能结果选择机制
- 📊 **详细输出**: 提供每个语言的识别结果和置信度信息
- 🔄 **自动优化**: 自动选择每个文本区域的最佳识别结果

## 模型配置详解

服务使用 `default_confs/config.yaml` 配置文件：

```yaml
detection:
  model_name: PP-OCRv3_mobile_det
recognition:
  fr:
    model_name: latin_PP-OCRv5_mobile_rec
  en:
    model_name: en_PP-OCRv5_mobile_rec
  ar:
    model_name: arabic_PP-OCRv3_mobile_rec
model_dir: models
```

### 模型作用说明

#### 检测模型 (Detection Model)
- **模型**: `PP-OCRv3_mobile_det`
- **作用**: 统一检测图像中的所有文本区域，生成文本块的边界框坐标
- **优势**: 确保所有语言使用相同的文本区域定位，避免不同语言模型检测结果不一致的问题

#### 识别模型 (Recognition Models)
- **法语模型**: `latin_PP-OCRv5_mobile_rec` - 专门优化拉丁字符识别
- **英语模型**: `en_PP-OCRv5_mobile_rec` - 针对英语文本优化的识别模型
- **阿拉伯语模型**: `arabic_PP-OCRv3_mobile_rec` - 专门处理阿拉伯文字的识别模型

每个识别模型都针对特定语言进行了优化，能够提供该语言的最佳识别效果。

### 配置生成机制
系统会根据配置自动生成 `running_confs/` 目录下的运行时配置文件，每个语言对应一个配置文件（如 `en.yaml`、`ar.yaml`、`fr.yaml`），确保每个OCR引擎使用正确的模型组合。

## 工作流程详解

### 1. 图像预处理
- 接收上传的图像文件
- 验证文件格式和大小
- 生成唯一的临时文件路径（使用UUID确保唯一性）

### 2. 统一文本检测阶段
```
输入图像 → PP-OCRv3_mobile_det → 文本区域坐标 (text blocks)
```
- 使用统一的检测模型 `PP-OCRv3_mobile_det` 识别图像中的所有文本区域
- 生成每个文本区域的边界框坐标 (bbox)
- 确保所有后续识别任务使用相同的文本区域定位

### 3. 多语言并行识别阶段
```
每个text block → [en_model, ar_model, fr_model] → 多语言识别结果
```
对于每个检测到的文本区域，系统会：
- 异步并行调用所有配置的语言识别模型
- 每个语言模型独立处理相同的文本区域
- 生成包含识别文本和置信度分数的结果

### 4. 置信度融合阶段
```python
# 核心算法：find_best_words_from_groups
for each_text_block:
    candidates = [en_result, ar_result, fr_result]
    best_result = max(candidates, key=lambda x: x.confidence_score)
    final_results.append(best_result)
```
- 对每个文本区域的多语言识别结果进行置信度比较
- 自动选择置信度最高的识别结果
- 保留所有语言的识别结果供参考

### 5. 结果组合与输出
- 将所有文本区域的最优识别结果组合
- 生成包含文本内容、坐标、置信度的完整输出
- 提供详细的多语言识别结果排名信息

## 性能优化

1. **并行处理**: 使用 `ThreadPoolExecutor` 和 `asyncio` 实现真正的并行处理
2. **线程池**: 默认使用4个工作线程，可根据服务器配置调整
3. **异步IO**: 所有网络操作都是异步的，不会阻塞其他请求

## API响应格式

### 主要输出结构
```json
{
  "language": "en,ar,fr",                    // 识别到的语言列表
  "bbox": {                                  // 文本区域坐标
    "0": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]],
    "1": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
  },
  "words_result": {                          // 最优识别结果
    "0": "Hello World",
    "1": "مرحبا بالعالم"
  },
  "words_num": 24,                           // 总字符数
  "best_scores": [0.95, 0.87],             // 每个区域的最高置信度
  "combined_text": "Hello World مرحبا بالعالم", // 组合文本
  "average_confidence": 0.91,                // 平均置信度
  "rank_reco_resp": [                        // 详细的多语言识别结果
    {
      "language": "en",
      "bbox": {...},
      "words_result": {...},
      "scores": [0.95, 0.82]
    },
    {
      "language": "ar", 
      "bbox": {...},
      "words_result": {...},
      "scores": [0.78, 0.87]
    }
  ]
}
```

### 响应字段说明
- **language**: 参与识别的所有语言
- **bbox**: 每个文本区域的四个角点坐标
- **words_result**: 经过置信度融合后的最优识别结果
- **best_scores**: 每个文本区域选中结果的置信度分数
- **combined_text**: 所有识别文本的组合结果
- **average_confidence**: 整体识别的平均置信度
- **rank_reco_resp**: 包含所有语言的详细识别结果，用于结果分析和调试

## 使用示例

### 基本调用
```bash
curl -X POST "http://localhost:80/" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@image.jpg"
```

### 指定语言(下个版本)
```bash
curl -X POST "http://localhost:80/" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@image.jpg" \
  -F "languages=en,ar"
```

## 支持的语言

当前支持的语言取决于 `config.yaml` 中配置的模型：

- `en`: 英语
- `ar`: 阿拉伯语  
- `fr`: 法语

可以通过添加更多PaddleOCR模型来扩展语言支持。

## 错误处理

- 不支持的文件格式会返回400错误
- 不支持的语言会在响应中说明
- 处理异常会返回详细的错误信息

## 部署建议

### 开发环境
```bash
# 直接运行
python api_server.py

# 或使用uvicorn
uvicorn api_server:app --host 0.0.0.0 --port 80 --reload
```

### 生产环境
```bash
# 多进程部署（推荐）
uvicorn api_server:app --host 0.0.0.0 --port 80 --workers 4

# 使用gunicorn
gunicorn api_server:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:80
```

### 性能调优建议
1. **Worker数量**: 建议设置为CPU核心数的1-2倍，考虑OCR模型的显存占用
2. **内存管理**: 每个worker会加载完整的OCR模型，需要充足的显存
3. **负载均衡**: 可以部署多个实例并使用负载均衡器
4. **日志管理**: 
   - 日志文件自动轮转（10MB/文件，保留10个备份）
   - 日志存储在 `logs/` 目录下
   - 支持不同级别的日志输出

### Docker部署
```dockerfile
FROM python:3.8-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 80

CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "80", "--workers", "4"]
```

## 技术特点总结

### 创新点
1. **统一检测架构**: 避免了不同语言模型检测结果不一致的问题
2. **智能置信度融合**: 自动选择最佳识别结果，无需人工干预
3. **异步并行处理**: 大幅提升多语言识别的处理效率
4. **完整结果保留**: 提供所有语言的识别结果，便于分析和调试

### 适用场景
- 多语言混合文档的OCR识别
- 不确定文本语言的自动识别
- 需要高精度识别结果的应用场景
- 大批量文档的自动化处理

### 扩展性
- 支持添加更多PaddleOCR语言模型
- 可以调整置信度融合算法
- 支持自定义检测和识别模型
- 易于集成到现有的文档处理流程中
=======
