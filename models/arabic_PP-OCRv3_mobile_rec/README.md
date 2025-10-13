---
license: Apache License 2.0
library_name: PaddleOCR
language:
- English
- Chinese
pipeline_tag: image-to-text
tags:
- OCR
- PaddlePaddle
- PaddleOCR
---

# arabic_PP-OCRv3_mobile_rec

## Introduction

arabic_PP-OCRv3_mobile_rec is a text line recognition model within the PP-OCRv3_rec series, developed by the PaddleOCR team. The arabic_PP-OCRv3_mobile_rec model is an Arabic-alphabet-specific model trained based on PP-OCRv3_mobile_rec, and it supports Arabic alphabet recognition. The key accuracy metrics are as follow:

<table>
<tr>
<th>Model</th>
<th>Recognition Avg Accuracy(%)</th>
<th>Model Storage Size (M)</th>
<th>Introduction</th>
</tr>
<tr>
<td>arabic_PP-OCRv3_mobile_rec</td>
<td>73.55</td>
<td>7.8 M</td>
<td>An ultra-lightweight Arabic alphabet recognition model trained based on the PP-OCRv3 recognition model, supporting Arabic alphabet and numeric character recognition.</td>
</tr>

</table>



**Note**: If any character (including punctuation) in a line was incorrect, the entire line was marked as wrong. This ensures higher accuracy in practical applications.

## Quick Start

### Installation

1. PaddlePaddle

Please refer to the following commands to install PaddlePaddle using pip:

```bash
# for CUDA11.8
python -m pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/

# for CUDA12.6
python -m pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/

# for CPU
python -m pip install paddlepaddle==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
```

For details about PaddlePaddle installation, please refer to the [PaddlePaddle official website](https://www.paddlepaddle.org.cn/en/install/quick).

2. PaddleOCR

Install the latest version of the PaddleOCR inference package from PyPI:

```bash
python -m pip install paddleocr
```

### Model Usage

You can quickly experience the functionality with a single command:

```bash
paddleocr text_recognition \
    --model_name arabic_PP-OCRv3_mobile_rec \
    -i https://cdn-uploads.huggingface.co/production/uploads/681c1ecd9539bdde5ae1733c/EPkFpN_xkKTh1URMOsjE6.png
```

You can also integrate the model inference of the text recognition module into your project. Before running the following code, please download the sample image to your local machine.

```python
from paddleocr import TextRecognition
model = TextRecognition(model_name="arabic_PP-OCRv3_mobile_rec")
output = model.predict(input="EPkFpN_xkKTh1URMOsjE6.png", batch_size=1)
for res in output:
    res.print()
    res.save_to_img(save_path="./output/")
    res.save_to_json(save_path="./output/res.json")
```

After running, the obtained result is as follows:

```json
{'res': {'input_path': '/root/.paddlex/predict_input/EPkFpN_xkKTh1URMOsjE6.png', 'page_index': None, 'rec_text': 'ددعتم يبرع صن رابتخا ةلاح', 'rec_score': 0.9411801695823669}}
```

For details about usage command and descriptions of parameters, please refer to the [Document](https://paddlepaddle.github.io/PaddleOCR/latest/en/version3.x/module_usage/text_recognition.html#iii-quick-start).

### Pipeline Usage

The ability of a single model is limited. But the pipeline consists of several models can provide more capacity to resolve difficult problems in real-world scenarios.

#### PP-OCRv3

The general OCR pipeline is used to solve text recognition tasks by extracting text information from images and outputting it in string format. And there are 5 modules in the pipeline: 
* Document Image Orientation Classification Module (Optional)
* Text Image Unwarping Module (Optional)
* Text Line Orientation Classification Module (Optional)
* Text Detection Module
* Text Recognition Module

Run a single command to quickly experience the OCR pipeline:

```bash
paddleocr ocr -i https://cdn-uploads.huggingface.co/production/uploads/681c1ecd9539bdde5ae1733c/Z1WF0PAh9lBnXZdOYE9QL.png \
    --text_recognition_model_name arabic_PP-OCRv3_mobile_rec \
    --use_doc_orientation_classify False \
    --use_doc_unwarping False \
    --use_textline_orientation True \
    --save_path ./output \
    --device gpu:0 
```

Results are printed to the terminal:

```json
{'res': {'input_path': '/root/.paddlex/predict_input/Z1WF0PAh9lBnXZdOYE9QL.png', 'page_index': None, 'model_settings': {'use_doc_preprocessor': True, 'use_textline_orientation': True}, 'doc_preprocessor_res': {'input_path': None, 'page_index': None, 'model_settings': {'use_doc_orientation_classify': False, 'use_doc_unwarping': False}, 'angle': -1}, 'dt_polys': array([[[  6,   5],
        ...,
        [  6,  38]],

       [[199,  32],
        ...,
        [200,  68]]], dtype=int16), 'text_det_params': {'limit_side_len': 64, 'limit_type': 'min', 'thresh': 0.3, 'max_side_limit': 4000, 'box_thresh': 0.6, 'unclip_ratio': 1.5}, 'text_type': 'general', 'textline_orientation_angles': array([0, 0]), 'text_rec_score_thresh': 0.0, 'rec_texts': ['ددعتم يبرع صن رابتخاةلاح', 'رطسألا'], 'rec_scores': array([0.98472452, 0.99996692]), 'rec_polys': array([[[  6,   5],
        ...,
        [  6,  38]],

       [[199,  32],
        ...,
        [200,  68]]], dtype=int16), 'rec_boxes': array([[  6, ...,  38],
       [199, ...,  68]], dtype=int16)}}
```

The command-line method is for quick experience. For project integration, also only a few codes are needed as well:

```python
from paddleocr import PaddleOCR  

ocr = PaddleOCR(
    text_recognition_model_name="arabic_PP-OCRv3_mobile_rec",
    use_doc_orientation_classify=False, # Use use_doc_orientation_classify to enable/disable document orientation classification model
    use_doc_unwarping=False, # Use use_doc_unwarping to enable/disable document unwarping module
    use_textline_orientation=True, # Use use_textline_orientation to enable/disable textline orientation classification model
    device="gpu:0", # Use device to specify GPU for model inference
)
result = ocr.predict("https://cdn-uploads.huggingface.co/production/uploads/681c1ecd9539bdde5ae1733c/Z1WF0PAh9lBnXZdOYE9QL.png")  
for res in result:  
    res.print()  
    res.save_to_img("output")  
    res.save_to_json("output")
```

The default model used in pipeline is `PP-OCRv5_server_rec`, so it is needed that specifing to `arabic_PP-OCRv3_mobile_rec` by argument `text_recognition_model_name`. And you can also use the local model file by argument `text_recognition_model_dir`. For details about usage command and descriptions of parameters, please refer to the [Document](https://paddlepaddle.github.io/PaddleOCR/latest/en/version3.x/pipeline_usage/OCR.html#2-quick-start).

## Links

[PaddleOCR Repo](https://github.com/paddlepaddle/paddleocr)

[PaddleOCR Documentation](https://paddlepaddle.github.io/PaddleOCR/latest/en/index.html)
