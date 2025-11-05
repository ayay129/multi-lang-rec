---
license: apache-2.0
library_name: PaddleOCR
language:
- en
- zh
pipeline_tag: image-to-text
tags:
- OCR
- PaddlePaddle
- PaddleOCR
- textline_detection
---

# PP-OCRv3_mobile_det

## Introduction

PP-OCRv3_mobile_det is one of the PP-OCRv3_det series models, a set of text detection models developed by the PaddleOCR team. This mobile-optimized text detection model offers higher efficiency, making it ideal for deployment on edge devices. 

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
paddleocr text_detection \
    --model_name PP-OCRv3_mobile_det \
    -i https://cdn-uploads.huggingface.co/production/uploads/681c1ecd9539bdde5ae1733c/3ul2Rq4Sk5Cn-l69D695U.png
```

You can also integrate the model inference of the text detection module into your project. Before running the following code, please download the sample image to your local machine.

```python
from paddleocr import TextDetection
model = TextDetection(model_name="PP-OCRv3_mobile_det")
output = model.predict(input="3ul2Rq4Sk5Cn-l69D695U.png", batch_size=1)
for res in output:
    res.print()
    res.save_to_img(save_path="./output/")
    res.save_to_json(save_path="./output/res.json")
```

After running, the obtained result is as follows:

```json
{'res': {'input_path': '/root/.paddlex/predict_input/3ul2Rq4Sk5Cn-l69D695U.png', 'page_index': None, 'dt_polys': array([[[ 637, 1429],
        ...,
        [ 634, 1450]],

       ...,

       [[ 356,  106],
        ...,
        [ 356,  127]]], dtype=int16), 'dt_scores': [0.8440782190003071, 0.7211973560197601, ..., 0.9473868156887905]}}
```

The visualized image is as follows:

![image/jpeg](https://cdn-uploads.huggingface.co/production/uploads/684ba591e717a30275a1b76a/ZjQKeKoqzjebOC7X-MEEc.png)

For details about usage command and descriptions of parameters, please refer to the [Document](https://paddlepaddle.github.io/PaddleOCR/latest/en/version3.x/module_usage/text_detection.html#iii-quick-start).

### Pipeline Usage

The ability of a single model is limited. But the pipeline consists of several models can provide more capacity to resolve difficult problems in real-world scenarios.

#### PP-OCRv3

The general OCR pipeline is used to solve text recognition tasks by extracting text information from images and outputting it in text form. And there are 5 modules in the pipeline: 
* Document Image Orientation Classification Module (Optional)
* Text Image Unwarping Module (Optional)
* Text Line Orientation Classification Module (Optional)
* Text Detection Module
* Text Recognition Module

Run a single command to quickly experience the OCR pipeline:

```bash
paddleocr ocr -i https://cdn-uploads.huggingface.co/production/uploads/681c1ecd9539bdde5ae1733c/3ul2Rq4Sk5Cn-l69D695U.png \
    --text_detection_model_name PP-OCRv3_mobile_det \
    --text_recognition_model_name PP-OCRv3_mobile_rec \
    --use_doc_orientation_classify False \
    --use_doc_unwarping False \
    --use_textline_orientation False \
    --save_path ./output \
    --device gpu:0 
```

Results are printed to the terminal:

```json
{'res': {'input_path': '/root/.paddlex/predict_input/3ul2Rq4Sk5Cn-l69D695U.png', 'page_index': None, 'model_settings': {'use_doc_preprocessor': True, 'use_textline_orientation': False}, 'doc_preprocessor_res': {'input_path': None, 'page_index': None, 'model_settings': {'use_doc_orientation_classify': False, 'use_doc_unwarping': False}, 'angle': -1}, 'dt_polys': array([[[ 354,  106],
        ...,
        [ 354,  127]],

       ...,

       [[ 633, 1433],
        ...,
        [ 633, 1449]]], dtype=int16), 'text_det_params': {'limit_side_len': 64, 'limit_type': 'min', 'thresh': 0.3, 'max_side_limit': 4000, 'box_thresh': 0.6, 'unclip_ratio': 1.5}, 'text_type': 'general', 'textline_orientation_angles': array([-1, ..., -1]), 'text_rec_score_thresh': 0.0, 'rec_texts': ['Algorithms for the Markov Entropy Decomposition', 'Andrew J.Ferris and David Poulin', 'Departement de Physique, Universite de Sherbrooke,Quebec, JIK 2RI, Canada', '(Dated: October 31,2018)', 'The Markov entropy decomposition (MED)is a recently-proposed, cluster-based simulation method for fi-', 'nite temperature quantum systems with arbitrary geometry. In this paper, we detail numerical algorithms for', 'performing the required steps of the MED,principally solving aminimization problem with a preconditioned', '09', "Newton's algorithm, aswell ashowtoextractglobal susceptibilities and thermal responses.Wedemonstrate", 'thepower of the method withthe spin-1/2XXZmodel on the 2D square lattice, including the extraction of', 'criticalpointsanddetailsofeachphase.Althoughthemethodsharessomequalitativesimilaritieswithexact-', 'diagonalization, we show theMEDisbothmore accurate and significantlymoreflexible.', 'PACS numbers: 05.10.a, 02.50.Ng, 03.67.a, 74.40.Kb', 'I.INTRODUCTION', 'This approximation becomes exact in the case of a1D quan-', 'tum (or classical) Markov chain [1O], and leads to an expo-', '[', 'Although the equations governing quantum many-body', 'nential reduction of costforexactentropy calculationswhen', 'systemsare simpleto write down,finding solutions for the', 'theglobaldensitymatrixis ahigher-dimensional Markovnet-', 'majority of systems remains incrediblydifficult.Modern', 'work state[12, 13].', 'physics finds itself in need of new tools to compute the emer-', 'The second approximation used in theMED approach is', 'gent behavior of large, many-body systems.', 'related to the N-representibilityproblem.Givena set of lo-', 'There has been a great variety of tools developed to tackle', 'cal but overlapping reduced density matrices fp:f, it is a very', 'many-bodyproblems,butingeneral,large2Dand3Dquan-', 'challengingproblem to determine if there exists aglobal den-', 'tum systems remain hard to deal with.Most systems are', 'sity operator which is positive semi-definite and whose partial', 'thought to be non-integrable,so exact analytic solutions are', 'trace agrees with each p. This problem is QMA-hard (the', 'notusuallyexpected.Directnumerical diagonalizationcanbe', 'quantumanalogue of NP)[14,15],and is hopelessly diffi-', 'performed for relatively small systemshowever the emer-', 'culttoenforce.Thus,thesecondapproximationemployed', 'gentbehavior of a system in thethermodynamic limitmaybe', 'involves ignoringglobal consistency withapositive opera-', 'difficult to extract, especially in systems with large correlation', 'tor,whilerequiringlocalconsistencyonanyoverlappingre-', 'lengths.MonteCarlo approaches aretechnically exact (up to', 'gions between the pi. At the zero-temperature limit, the MED', '', 'sampling error),but sufferfrom the so-called sign problem', 'approach becomes analogous tothe variational nth-order re-', '一', 'forfermionic,frustrated,or dynamicalproblems.Thus we are', 'duced density matrix approach, where positivity is enforced', '', 'limited to search for clever approximations to solve the ma-', 'onallreduceddensitymatricesofsizen[16-18].', 'jorityofmany-bodyproblems.', 'The MED approachis an extremely flexible cluster method,', 'Over the past century,hundreds of such approximations', 'applicabletobothtranslationally invariant systems of anydi-', 'have been proposed, and we will mention just a few notable', 'mensioninthethermodynamiclimit,aswell asfinite systems', '1', 'examples applicable to quantumlattice models.Mean-field', 'or systems without translational invariance (e.g. disordered', 'theory is simple and frequently arrives at the correct quali-', 'lattices,orharmonicallytrapped atoms in optical lattices)', '11', 'tativedescription,butoftenfails when correlations areim-', 'Thefree energy given byMED is guaranteed to lowerbound', 'portant.Density-matrix renormalisation group (DMRG) [1]', 'the true free energy, which in turn lower-bounds the ground', '[ :A!', 'is efficient and extremely accurate at solving 1D problems', 'state energy-thus providing a natural complement to varia-', 'but the computational cost grows exponentially with system', 'tional approaches which upper-bound the ground state energy.', '!XIe', 'size in two- or higher-dimensions [2, 3].Related tensor-', 'The ability to provide a rigorous ground-state energy window', 'networktechniquesdesignedfor2Dsystemsarestillintheir', 'is a powerful validation tool, creating a very compellingrea-', 'infancy[4-6].Series-expansionmethods[7]canbe success-', 'son to use this approach.', 'ful, but may diverge or otherwise converge slowly, obscuring', 'In this paper we paper we present a pedagogical introduc-', 'the state in certain regimes. There exist a variety of cluster-', 'tion to MED, including numerical implementation issues and', 'based techniques, such as dynamical-mean-field theory[8]', 'applicationsto 2D quantumlatticemodels in thethermody-', 'and density-matrix embedding [9].', 'namiclimit.In Sec.II,we give a brief derivation of the', 'Herewediscusstheso-calledMarkoventropydecompo-', 'Markov entropydecomposition.SectionIIIoutlinesarobust', 'sition (MED),recently proposed by Poulin & Hastings [1O]', 'numerical strategy for optimizing the clusters that make up', '(andanalogoustoaslightlyearlierclassicalalgorithm[11)).', 'thedecomposition.InSec.IVweshowhowwecanextend', 'This is a self-consistent cluster method for finitetemperature', 'these algorithms toextractnon-trivial information,such as', 'systemsthattakesadvantageofanapproximationofthe(von', 'specific heat and susceptibilities. We present an application of', 'Neumann) entropy. In [1o], it was shown that the entropy', 'the method to the spin-1/2 XXZ model on a 2D square lattice', 'persitecanberigorouslyupperboundedusingonlylocalin-', 'inSec.V,describinghowtocharacterizethephasediagram', 'formationa local,reduced densitymatrix onN sites,say.', 'anddeterminecriticalpoints,beforeconcludinginSec.Vl'], 'rec_scores': array([0.92904288, ..., 0.92923349]), 'rec_polys': array([[[ 354,  106],
        ...,
        [ 354,  127]],

       ...,

       [[ 633, 1433],
        ...,
        [ 633, 1449]]], dtype=int16), 'rec_boxes': array([[ 354, ...,  128],
       ...,
       [ 633, ..., 1449]], dtype=int16)}}
```

If save_path is specified, the visualization results will be saved under `save_path`. The visualization output is shown below:

![image/jpeg](https://cdn-uploads.huggingface.co/production/uploads/684ba591e717a30275a1b76a/FWtokrOWWSD-MXhyIJ1fr.png)

The command-line method is for quick experience. For project integration, also only a few codes are needed as well:

```python
from paddleocr import PaddleOCR  

ocr = PaddleOCR(
    text_detection_model_name="PP-OCRv3_mobile_det",
    text_recognition_model_name="PP-OCRv3_mobile_rec",
    use_doc_orientation_classify=False, # Disables document orientation classification model via this parameter
    use_doc_unwarping=False, # Disables text image rectification model via this parameter
    use_textline_orientation=False, # Disables text line orientation classification model via this parameter
)
result = ocr.predict("./3ul2Rq4Sk5Cn-l69D695U.png")  
for res in result:  
    res.print()  
    res.save_to_img("output")  
    res.save_to_json("output")
```

For details about usage command and descriptions of parameters, please refer to the [Document](https://paddlepaddle.github.io/PaddleOCR/latest/en/version3.x/pipeline_usage/OCR.html#2-quick-start).


## Links

[PaddleOCR Repo](https://github.com/paddlepaddle/paddleocr)

[PaddleOCR Documentation](https://paddlepaddle.github.io/PaddleOCR/latest/en/index.html)