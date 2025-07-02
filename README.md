# Scan-and-Print: Patch-level Data Summarization and Augmentation for Content-aware Layout Generation in Poster Design

This repository contains the Pytorch implementation for "[Scan-and-Print: Patch-level Data Summarization and Augmentation for Content-aware Layout Generation in Poster Design](https://arxiv.org/abs/2505.20649)", IJCAI 2025 (AI, Arts and Creativity).

<img src="/topic.jpg" alt="Scan-and-Print">
<p align="center">What's new in Scan-and-Print.</p>

## How to Run

### Prerequisites
- Environment
```
Python 3.10.13
CUDA 12.2
```
- Main Modules
```
torch==2.1.1
torchvision==0.16.1
timm==0.9.7
opencv-python==4.8.1.78
pandas==2.2.3
Pillow==10.0.1
segmentation-models-pytorch==0.3.4
numpy==1.26.2
```
- Other Modules: Please refer to ```requirements.txt```

### Preparation
1. Data Preparation
- Download the [PKU PosterLayout dataset](https://github.com/PKU-ICST-MIPL/PosterLayout-CVPR2023) and [CGL dataset](https://github.com/minzhouGithub/CGL-GAN) from their official websites. Only for convenience, we provide the [processed annotation files](https://drive.google.com/drive/folders/1GGh02Zv0sDjTai3FE0uNPntm-Asj8ioD?usp=sharing) with a uniform csv format. Please make sure to obtain the corresponding agreement before use.
- Follow the instructions of [RALF](https://github.com/CyberAgentAILab/RALF) to preprocess the images, including:
    - ```inpainting.py```: results should be put under ```input``` directory.
    - ```saliency_detection.py```: results should be put under ```saliency``` and ```saliency_sub``` directories.
- The file structure should be as follows:
```
├── AbsolutePath/to/DatasetDirectory
│   ├── pku
│   │   ├── annotation
│   │   │   ├── test.csv
│   │   │   └── train.csv
│   │   └── image
│   │       ├── test
│   │       │   ├── input
│   │       │   ├── saliency
│   │       │   └── saliency_sub
│   │       └── train
│   │           ├── input
│   │           ├── original
│   │           ├── saliency
│   │           └── saliency_sub
│   └── cgl
│       ├── annotation
│       │   └── ... (as above)
│       └── image
│           └── ... (as above)
└── AbsolutePath/to/Scan-and-Print
```

2. Global Variables Setting
- Specify AbsolutePath/to/DatasetDirectory and AbsolutePath/to/Scan-and-Print in the ```init_path.sh``` file and execute the following command.
```
source init_path.sh
```

3. Density (Design Intent) Detection
- The **design_intent_detect** directory is mostly identical to the one in [PosterO-CVPR2025](https://github.com/theKinsley/PosterO-CVPR2025), but there are two differences:
  - Some hyperparameters are different for training on individual dataset. Check the ```train.sh``` script for details.
  - Only density maps are obtained during inference. Check the ```test.sh``` script for details.
- Download the [weights](https://drive.google.com/drive/folders/1-h-6aLzphktW6gXizEjzecBNiyII1BxX?usp=sharing) of the detection model or train it from scratch with ```source train.sh <DATASET>```.
- The file structure should be as follows:
```
AbsolutePath/to/Scan-and-Print
└── design_intent_detect
    ├── pku_64_0.0001_none_conloss
    │   └── ckpt/pku_epoch70.pth
    ├── cgl_128_0.0001_none_conloss
    │   └── ckpt/cgl_epoch25.pth
    └── ...
```
- Execute the following commands to obtain the detection results. Noted that GPU IDs should be specified in ```test.sh```
```
cd design_intent_detect
source test.sh <DATASET> <PATH_TO_WEIGHT>
```
For example, ```source test.sh pku pku_64_0.0001_none_conloss/ckpt/pku_epoch70.pth``` for the PKU PosterLayout dataset.

3. Symbolic Linking
- Create symbolic links to the detection results in the AbsolutePath/to/DatasetDirectory. The file structure should be as follows:
```
├── AbsolutePath/to/DatasetDirectory
│   ├── pku
│   │   ├── ...
│   │   └── image
│   │       ├── test
│   │       │   ├── ...
│   │       │   └── density -> *AbsolutePath/to/Scan-and-Print/design_intent_detect/pku_64_0.0001_none_conloss/result/pku_epoch70/test*
│   │       └── train
│   │           ├── ...
│   │           └── density -> *AbsolutePath/to/Scan-and-Print/design_intent_detect/pku_64_0.0001_none_conloss/result/pku_epoch70/train*
│   └── cgl
│       ├── ...
│       └── image
│           └── ... (as above, but link to the corresponding density maps)
└── AbsolutePath/to/Scan-and-Print
```

### Layout Generation
1. Ensure the working directory is **AbsolutePath/to/Scan-and-Print**.

2. Train
- Download the pretrained [weights](https://drive.google.com/drive/folders/1-h-6aLzphktW6gXizEjzecBNiyII1BxX?usp=sharing) of DeiT3 and place it under the ```cache``` directory.
- Execute the following commands to start training.
```
source script/train.sh <GPU_ID> <DATASET>
```
For example, ```source script/train.sh 0 pku``` for the PKU PosterLayout dataset. The experiment directory will be saved at ```Scan-and-Print/history/filtering_<timestamp>/<DATASET>```.

3. Inference and Evaluation
- For baseline comparison, execute the following command to do testing and evaluation.
```
source script/test.sh <GPU_ID> <PATH_TO_EXPERIMENT_DIRECTORY(IES)>
source script/eval.sh <PATH_TO_EXPERIMENT_DIRECTORY(IES)>
```
- For **C->S+P** constrained generation task (no extra training is needed!), execute the following command to do testing and evaluation.
```
source script/test_.sh <GPU_ID> <PATH_TO_EXPERIMENT_DIRECTORY(IES)>
source script/eval.sh <PATH_TO_EXPERIMENT_DIRECTORY(IES)>
```
- For visualize the generated layouts, execute the following command. Images will be saved at ```<PATH_TO_EXPERIMENT_DIRECTORY>/eval(_c)/visualize```.
```
# unconditional task
source script/visualize.sh <PATH_TO_EXPERIMENT_DIRECTORY>
# C->S+P constrained task
source script/visualize.sh <PATH_TO_EXPERIMENT_DIRECTORY> c
```

## Citation
If our work is helpful for your research, please cite our papers:
```
@inproceedings{Hsu-IJCAI2025-ScanandPrint,
  title={Scan-and-Print: Patch-level Data Summarization and Augmentation for Content-aware Layout Generation in Poster Design},
  author={Hsu, HsiaoYuan and Peng, Yuxin},
  booktitle={Proceedings of the International Joint Conference on Artificial Intelligence},
  year={2025}
}
```

## Contact me
For any questions or further information, please feel free to reach me with email kslh99@outlook.com🫡