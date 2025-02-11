# Crop Classification In UAV Orthophotos Using Deep Learning For Hilly Region of Nepal

This repository contains code for training and evaluating a UNet model for multi-class image segmentation.  
The **main script** is **`unet_final_3.py`** — all functionality and workflows are contained there.



![Banepa Tunnel Feasibility Survey_transparent_mosaic_group19352](https://github.com/user-attachments/assets/11b5dfdd-cd95-44ba-a1f0-b4e21c6bb507)
![Banepa Tunnel Feasibility Survey_transparent_mosaic_group116585](https://github.com/user-attachments/assets/4cf3143a-cf48-4fd8-82ff-3f62e9568555)
![Banepa Tunnel Feasibility Survey_transparent_mosaic_group126826](https://github.com/user-attachments/assets/f3e56800-6ce0-46e9-9eaa-d7dfa5b6238f)

f1_score:88.65 iou_score: 65.39

## Table of Contents
1. [Features](#features)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Project Structure](#project-structure)
5. [Model Description](#model-description)
6. [Dataset Preparation](#dataset-preparation)
7. [Training & Validation Process](#training--validation-process)
8. [Monitoring Training](#monitoring-training)
9. [Results and Inference](#results-and-inference)


## Features
- **UNet Architecture**: Customizable final layer for multi-class segmentation (4 classes).
- **Training & Validation**:
  - IoU & F1 Score metrics on the fly.
  - Early stopping based on validation IoU.
- **Logging**:
  - TensorBoard logging with `SummaryWriter`.
  - CSV logging for numeric summaries (`summary.csv`).
- **Visualization**:
  - Prediction vs ground truth mask plotting.
  - Color-coded segmentation masks.

## Installation
```
pip install torch torchvision torchaudio
pip install torchmetrics
pip install numpy matplotlib progressbar
pip install tensorboard
```



## Usage
```
DATASET_PATH = "/path/to/your/dataset"
TRAIN_IMAGES_DIR = "/path/to/your/dataset/train/images"
TRAIN_MASKS_DIR = "/path/to/your/dataset/train/masks"
VAL_IMAGES_DIR   = "/path/to/your/dataset/val/images"
VAL_MASKS_DIR   = "/path/to/your/dataset/val/masks"
```
## Run Training
```
python unet_final_3.py --epochs 50 --batch_size 4
```
## Monitor Training with TensorBoard
```
tensorboard --logdir runs
```

## Inference / Evaluation
```
python unet_final_3.py --eval --model_checkpoint ./checkpoints/best_model.pth

```

## Model Description
The model is a UNet architecture adapted from mateuszbuda/brain-segmentation-pytorch.
Key modifications:

The final layer has 4 output channels for segmentation into 4 classes.
Uses Dice Loss or Cross-Entropy Loss for optimization.

## Dataset Preparation
```
/dataset
   ├── train
   │   ├── images
   │   └── masks
   ├── val
   │   ├── images
   │   └── masks
   ├── test (optional)
```
## Training graph
![training](https://github.com/user-attachments/assets/7111f75f-c8f9-468f-9492-f141c2ce7251)





