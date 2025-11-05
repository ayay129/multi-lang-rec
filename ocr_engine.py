import os
from paddleocr import PaddleOCR
import yaml
import copy
default_confs_dir = os.path.join(os.path.dirname(__file__), 'default_confs')
running_confs_dir = os.path.join(os.path.dirname(__file__), 'running_confs')
config_dir = os.path.join(default_confs_dir, 'config.yaml')
default_config = os.path.join(default_confs_dir, 'DefaultOCR.yaml')

def init_models():
    with open(config_dir, 'r') as file:
        config = yaml.safe_load(file)
    with open(default_config, 'r') as file:
        default_pd_config = yaml.safe_load(file)
    model_dir = config["model_dir"]
    det_name = config['detection']['model_name']
    rec_config = config['recognition']
    ocr_engines = {}
    for lang, rec in rec_config.items():
        lang_config_file = os.path.join(running_confs_dir, f'{lang}.yaml')
        if not os.path.exists(lang_config_file):
            lang_config = copy.deepcopy(default_pd_config)
            # 修改TextDetection配置
            lang_config['SubModules']['TextDetection']['model_dir'] = f"{model_dir}/{det_name}"
            lang_config['SubModules']['TextDetection']['model_name'] = det_name
        
            # 修改TextRecognition配置
            rec_model_name = rec['model_name']
            lang_config['SubModules']['TextRecognition']['model_dir'] = f"{model_dir}/{rec_model_name}"
            lang_config['SubModules']['TextRecognition']['model_name'] = rec_model_name
            with open(lang_config_file, 'w') as f:
                yaml.dump(lang_config, f, default_flow_style=False,allow_unicode=True,indent=2)
        ocr_engines[lang] = PaddleOCR(paddlex_config=lang_config_file)
    return ocr_engines


ocr_engines = init_models()

    