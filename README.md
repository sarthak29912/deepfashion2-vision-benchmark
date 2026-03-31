# DeepFashion2 Vision Benchmark

## Overview

This project focuses on fashion image understanding using a pruned version of the DeepFashion2 dataset. The work covers three major computer vision tasks:

- multi-label image classification
- object detection
- image segmentation

The project was designed around five selected clothing categories and compares multiple deep learning models across transfer learning and training-from-scratch settings.

---

## Selected Clothing Categories

The following DeepFashion2 category IDs were used throughout the project:

- `1` → short sleeve top
- `8` → long sleeve top
- `7` → shorts
- `2` → trousers
- `9` → skirt

### Classification / YOLO label mapping

- `0` → short sleeve top
- `1` → long sleeve top
- `2` → shorts
- `3` → trousers
- `4` → skirt

### U-Net / Mask R-CNN label mapping

- `0` → background
- `1` → short sleeve top
- `2` → long sleeve top
- `3` → shorts
- `4` → trousers
- `5` → skirt

---

## Objectives

The project was divided into three parts:

### 1. Multi-label Classification
Predict all clothing categories present in a single image.

### 2. Detection
Detect clothing objects and localize them using bounding boxes.

### 3. Segmentation
Segment clothing regions at the pixel level for the selected categories.

---

## Models Used

### Classification
- ResNet50
- EfficientNet-B0
- MobileNetV3

Each classification model was trained using:
- transfer learning
- training from scratch

### Detection
- YOLO
- Mask R-CNN

### Segmentation
- U-Net

For segmentation and instance-based tasks, polygon annotations from DeepFashion2 were converted into masks.

---

## Training Approach

### Classification setup
The classification task was handled as a multi-label problem.

Main choices:
- BCEWithLogitsLoss
- Adam optimizer
- weighted loss using class imbalance correction
- macro and micro evaluation metrics

### Detection setup
For object detection:
- YOLO was trained on YOLO-format labels
- Mask R-CNN was trained using bounding boxes and masks extracted from annotation files

### Segmentation setup
For U-Net:
- semantic masks were created from polygon annotations
- the model predicted class-wise segmentation maps
- background was treated as a separate class

---

## What Has Been Done So Far

### Classification
- prepared the pruned 5-class label mapping
- built multi-label dataset pipelines
- trained ResNet50, EfficientNet-B0, and MobileNetV3
- ran transfer learning experiments
- ran scratch training experiments with Kaggle-safe adjustments
- evaluated transfer models on the validation set

### Detection
- prepared YOLO-compatible data
- trained YOLO transfer and scratch runs
- implemented Mask R-CNN training with Kaggle-safe subset strategy

### Segmentation
- built U-Net mask generation pipeline
- trained U-Net transfer model
- created checkpoint-based training flow for long Kaggle sessions

### Evaluation
- implemented classification evaluation
- implemented combined evaluation for detection and segmentation models
- compared transfer models across the selected classes

---

## Key Classification Result

Among the evaluated transfer-learning classification models, **ResNet50 Transfer** performed best overall on the validation set.

It achieved the strongest macro and micro F1-score among:
- ResNet50 Transfer
- EfficientNet-B0 Transfer
- MobileNetV3 Transfer

The most difficult classes were generally:
- shorts
- trousers

The more stable classes were:
- short sleeve top
- long sleeve top

---

## Practical Notes

This work was developed primarily on **Kaggle GPU**.

Because some models were computationally expensive, practical changes were necessary:
- reduced batch sizes
- gradient accumulation
- subset-based training for heavy models
- smaller image sizes where required
- epoch-wise checkpoint saving

These changes made it possible to run large-model experiments within Kaggle limits.

---

## Evaluation Metrics

### Classification
- Precision
- Recall
- F1-score
- Macro average
- Micro average

### Detection
- Precision
- Recall
- mAP@50
- mAP@50:95

### Segmentation
- mIoU
- Dice score

---

## Dataset Note

This project uses a pruned version of DeepFashion2 and restricts all experiments to only five selected classes. The annotations were parsed to build:
- multi-label targets for classification
- bounding box labels for detection
- mask labels for segmentation

---

## Future Work

Possible next improvements include:
- stronger hyperparameter tuning
- full COCO-style Mask R-CNN evaluation
- better post-processing of U-Net predictions into instance-level outputs
- more robust class balancing
- improved visualizations and report figures

---

## Summary

This project benchmarks multiple deep learning approaches for fashion understanding using a common 5-class subset of DeepFashion2. It combines classification, detection, and segmentation into one unified workflow and compares both pretrained and scratch-based training strategies under practical Kaggle constraints.
