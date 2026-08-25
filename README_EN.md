# Multi-language OCR FastAPI Service

**Language:** [中文](README.md) | **English** | [العربية](README_AR.md)

A multi-language precise recognition architecture based on PaddleOCR PPStructure pipeline logic, achieving high-precision multi-language text recognition through unified text detection + multi-language recognition + confidence fusion.

## Architecture Design

### Core Concept
This project is based on PaddleOCR's PPStructure pipeline logic, adopting an innovative architecture of "unified detection + language-specific recognition + confidence fusion":

1. **Unified Text Detection**: Use a single detection model to identify all text regions (text blocks) in the image
2. **Multi-language Parallel Recognition**: Send each text region to different language recognition models for processing
3. **Confidence Fusion**: Compare confidence scores of multi-language recognition results for each text region and select the optimal result
4. **Result Combination**: Combine the optimal recognition results of all text regions into the final output

### Technical Advantages
- 🎯 **Precise Positioning**: Unified detection ensures consistency and accuracy of text region positioning
- 🚀 **Parallel Processing**: Multi-language recognition tasks execute asynchronously in parallel, significantly improving processing efficiency
- 🧠 **Intelligent Fusion**: Confidence-based result selection automatically chooses the best recognition results
- 🌍 **Language Agnostic**: No need to pre-determine text language, system automatically selects optimal recognition results

## Features

- 🚀 **Asynchronous Parallel Processing**: Simultaneous OCR recognition for multiple languages, improving processing efficiency
- 🌍 **Multi-language Support**: Supports English (en), Arabic (ar), French (fr), and other languages
- 🎯 **Precise Recognition**: Intelligent result selection mechanism based on confidence scores
- 📊 **Detailed Output**: Provides recognition results and confidence information for each language
- 🔄 **Automatic Optimization**: Automatically selects the best recognition result for each text region

## Model Configuration Details

The service uses the `default_confs/config.yaml` configuration file:

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

### Model Function Description

#### Detection Model
- **Model**: `PP-OCRv3_mobile_det`
- **Function**: Uniformly detect all text regions in the image and generate bounding box coordinates for text blocks
- **Advantage**: Ensures all languages use the same text region positioning, avoiding inconsistent detection results from different language models

#### Recognition Models
- **French Model**: `latin_PP-OCRv5_mobile_rec` - Specially optimized for Latin character recognition
- **English Model**: `en_PP-OCRv5_mobile_rec` - Recognition model optimized for English text
- **Arabic Model**: `arabic_PP-OCRv3_mobile_rec` - Specialized for Arabic text recognition

Each recognition model is optimized for specific languages and can provide the best recognition performance for that language.

### Configuration Generation Mechanism
The system automatically generates runtime configuration files in the `running_confs/` directory based on the configuration, with each language corresponding to a configuration file (such as `en.yaml`, `ar.yaml`, `fr.yaml`), ensuring each OCR engine uses the correct model combination.

## Workflow Details

### 1. Image Preprocessing
- Receive uploaded image files
- Validate file format and size
- Generate unique temporary file paths (using UUID to ensure uniqueness)

### 2. Unified Text Detection Stage
```
Input Image → PP-OCRv3_mobile_det → Text Region Coordinates (text blocks)
```
- Use the unified detection model `PP-OCRv3_mobile_det` to identify all text regions in the image
- Generate bounding box coordinates (bbox) for each text region
- Ensure all subsequent recognition tasks use the same text region positioning

### 3. Multi-language Parallel Recognition Stage
```
Each text block → [en_model, ar_model, fr_model] → Multi-language Recognition Results
```
For each detected text region, the system will:
- Asynchronously call all configured language recognition models in parallel
- Each language model independently processes the same text region
- Generate results containing recognized text and confidence scores

### 4. Confidence Fusion Stage
```python
# Core algorithm: find_best_words_from_groups
for each_text_block:
    candidates = [en_result, ar_result, fr_result]
    best_result = max(candidates, key=lambda x: x.confidence_score)
    final_results.append(best_result)
```
- Compare confidence scores of multi-language recognition results for each text region
- Automatically select the recognition result with the highest confidence
- Retain recognition results from all languages for reference

### 5. Result Combination and Output
- Combine optimal recognition results from all text regions
- Generate complete output containing text content, coordinates, and confidence scores
- Provide detailed multi-language recognition result ranking information

## Performance Optimization

1. **Parallel Processing**: Use `ThreadPoolExecutor` and `asyncio` to achieve true parallel processing
2. **Thread Pool**: Default to 4 worker threads, adjustable based on server configuration
3. **Asynchronous IO**: All network operations are asynchronous and won't block other requests

## API Response Format

### Main Output Structure
```json
{
  "language": "en,ar,fr",                    // List of recognized languages
  "bbox": {                                  // Text region coordinates
    "0": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]],
    "1": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
  },
  "words_result": {                          // Optimal recognition results
    "0": "Hello World",
    "1": "مرحبا بالعالم"
  },
  "words_num": 24,                           // Total character count
  "best_scores": [0.95, 0.87],             // Highest confidence for each region
  "combined_text": "Hello World مرحبا بالعالم", // Combined text
  "average_confidence": 0.91,                // Average confidence
  "rank_reco_resp": [                        // Detailed multi-language recognition results
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

### Response Field Description
- **language**: All languages participating in recognition
- **bbox**: Four corner coordinates of each text region
- **words_result**: Optimal recognition results after confidence fusion
- **best_scores**: Confidence scores of selected results for each text region
- **combined_text**: Combined results of all recognized text
- **average_confidence**: Overall average confidence of recognition
- **rank_reco_resp**: Contains detailed recognition results for all languages, used for result analysis and debugging

## Usage Examples

### Basic Call
```bash
curl -X POST "http://localhost:80/" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@image.jpg"
```

### Specify Languages (Next Version)
```bash
curl -X POST "http://localhost:80/" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@image.jpg" \
  -F "languages=en,ar"
```

## Supported Languages

Currently supported languages depend on models configured in `config.yaml`:

- `en`: English
- `ar`: Arabic  
- `fr`: French

Language support can be extended by adding more PaddleOCR models.

## Error Handling

- Unsupported file formats will return 400 errors
- Unsupported languages will be indicated in the response
- Processing exceptions will return detailed error information

## Deployment Recommendations

### Development Environment
```bash
# Direct execution
python api_server.py

# Or use uvicorn
uvicorn api_server:app --host 0.0.0.0 --port 80 --reload
```

### Production Environment
```bash
# Multi-process deployment (recommended)
uvicorn api_server:app --host 0.0.0.0 --port 80 --workers 4

# Using gunicorn
gunicorn api_server:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:80
```

### Performance Tuning Recommendations
1. **Worker Count**: Recommended to set to 1-2x CPU cores, considering OCR model memory usage
2. **Memory Management**: Each worker loads complete OCR models, requiring sufficient memory
3. **Load Balancing**: Can deploy multiple instances and use load balancers
4. **Log Management**: 
   - Automatic log file rotation (10MB/file, retain 10 backups)
   - Logs stored in `logs/` directory
   - Support for different log levels

### Docker Deployment
```dockerfile
FROM python:3.8-slim

WORKDIR /app
COPY requirements*.txt ./
RUN pip install -r requirements.nvidia.txt

COPY . .
EXPOSE 80

CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "80", "--workers", "4"]
```

## Technical Features Summary

### Innovation Points
1. **Unified Detection Architecture**: Avoids inconsistent detection results from different language models
2. **Intelligent Confidence Fusion**: Automatically selects best recognition results without manual intervention
3. **Asynchronous Parallel Processing**: Significantly improves multi-language recognition processing efficiency
4. **Complete Result Retention**: Provides recognition results from all languages for analysis and debugging

### Application Scenarios
- OCR recognition of multi-language mixed documents
- Automatic recognition of texts with uncertain languages
- Applications requiring high-precision recognition results
- Automated processing of large document batches

### Extensibility
- Support for adding more PaddleOCR language models
- Adjustable confidence fusion algorithms
- Support for custom detection and recognition models
- Easy integration into existing document processing workflows
