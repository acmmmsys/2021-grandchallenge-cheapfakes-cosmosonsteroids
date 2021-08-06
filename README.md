## [COSMOS on Steroids: a Cheap Detector for Cheapfakes](https://github.com/acmmmsys/2021-grandchallenge-cheapfakes-cosmosonsteroids)

COSMOS dataset consists of images and captions scraped from news articles and other websites designed for training and evaluation of out-of-context use of images. We refer the readers to the paper below for more details. 

**Tankut Akgul, Tugce Erkilic Civelek, Deniz Ugur, and Ali C. Begen, "COSMOS on Steroids: a Cheap Detector for Cheapfakes," in ACM Multimedia Systems Conference (MMSys’21), Sept. 2021, Istanbul, Turkey (DOI: 10.1145/3458305. 3479968)**

## Getting started
1. **Source Code:**   `$ git clone` this repo and install the Python dependencies from `requirements.txt`. The source code is implemented in PyTorch so familarity with PyTorch is expected.

2. **Dataset:** Download the dataset by filling out the form [here](https://docs.google.com/forms/d/13kJQ2wlv7sxyXoaM1Ddon6Nq7dUJY_oftl-6xzwTGow). 
3. **Visualize Dataset:** It is difficult to view the dataset using only JSON file. Navigate to the directory `dataset_visualizer` and follow the instructions to visualize the dataset using a simple Python-Flask based web tool 
4. **Train and Test For Image-Text Matching Task:** This code is based on [Detectron2](https://github.com/facebookresearch/detectron2) to extract features from objects present in the image. Please setup and install detectron2 first if you wish to use our feature detector for images. The minimal changes to be done to detectron2 source code to extract object features are added to [detectron2_changes](detectron2_changes/) directory. Navigate to detectron2 source code directory and simply run `patch -p1 ../detectron2 < 0001-detectron2-mod.patch`. Consider setting up detectron2 inside this directory, it worked seamlessly for me without doing many changes.                                 
All the training parameters are configured via `utils/config.py`. Specify hyperparameters, text-embeddings, threshold values, etc in the [config .py](utils/config.py) file. Model names are specifed in the trainer script itself. Configure these parameters according to your need and start training.     
To train the model, execute the following command:
`python trainer_scipt.py -m train`      
Once training is finished, then to evaluate the model with Match vs No-Match Accuracy, execute the following command:
`python trainer_scipt.py -m eval`


## Evaluating
**Test For Out-of-Context Detection Accuracy:**  Once training is over, then to evaluate the model for out-of-Context Detection task, specify model name in `evaluate_ooc.py`.

In order to reproduce our results execaute the following commands for each section.

- **Section 3.1**: Differential Sensing
```bash
    export COSMOS_IOU=0.5
    export COSMOS_DISABLE_ISFAKE=1
    export COSMOS_DISABLE_RECT_OPTIM=1

    python evaluate_ooc.py
    
    unset $(env | sed -n 's/^\(COSMOS.*\)=.*/\1/p')
```

- **Section 3.2**: Fake-or-Fact
```bash
    export COSMOS_IOU=0.5
    export COSMOS_DISABLE_ISOPPOSITE=1
    export COSMOS_DISABLE_RECT_OPTIM=1

    python evaluate_ooc.py
    
    unset $(env | sed -n 's/^\(COSMOS.*\)=.*/\1/p')
```

- **Section 3.3**: Object-Caption Matching
```bash
    export COSMOS_IOU=0.5
    export COSMOS_WORD_DISABLE=1

    python evaluate_ooc.py
    
    unset $(env | sed -n 's/^\(COSMOS.*\)=.*/\1/p')
```

- **Section 3.4**: IoU Threshold Adjustment
```bash
    export COSMOS_IOU=0.25
    export COSMOS_WORD_DISABLE=1
    export COSMOS_DISABLE_RECT_OPTIM=1

    python evaluate_ooc.py
    
    unset $(env | sed -n 's/^\(COSMOS.*\)=.*/\1/p')
```

- **Section 3.1 + 3.2 + 3.3 + 3.4**: Proposed Method
```bash
    python evaluate_ooc.py
```

## Environment Variables
These variables modify the behaviour of our evaluation method

### Method specific variables
| Variable Name                 | Description                                                      |
|-------------------------------|------------------------------------------------------------------|
| **COSMOS_DISABLE_ISOPPOSITE** | Disables Section 3.1 Differential Sensing.                       |
| **COSMOS_DISABLE_ISFAKE**     | Disables Section 3.2 Fake-or-Fact.                               |
| **COSMOS_DISABLE_RECT_OPTIM** | Disables Section 3.3 Object-Caption Matching.                    |
| **COSMOS_IOU**                | Setting it to "0.5" disabled Section 3.4, and "0.25" enables it. |
| **COSMOS_WORD_DISABLE**       | Disables Section 3.1 and 3.2 altogether.                         |

### Dataset specific variables
| Variable Name                      | Description      |
|------------------------------------|------------------|
| **COSMOS_BASE_DIR**                | Base directory.  |
| **COSMOS_DATA_DIR** *(optional)*   | Data directory.  |
| **COSMOS_TARGET_DIR** *(optional)* | Target directory |

### Comparison specific variables
| Variable Name            | Description                                                       |
|--------------------------|-------------------------------------------------------------------|
| **COSMOS_COMPARE**       | Setting this to "1" will enable comparison with original paper.   |
| **COSMOS_COMPARE_LEVEL** | Choose between [0, 1, 2] (0 is default). Changes verbosity level. |


</br>
