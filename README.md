


---

# Off-Road Terrain Perception System

### AI Powered Semantic Segmentation for Autonomous Terrain Understanding

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-orange)
![CUDA](https://img.shields.io/badge/CUDA-GPU-green)
![Status](https://img.shields.io/badge/Project-Hackathon-purple)

An AI powered perception system designed to interpret **off-road environments** using semantic segmentation. The system classifies terrain types such as trees, rocks, grass, bushes, and landscape at the **pixel level**, enabling autonomous systems to understand and navigate unstructured environments.

This project demonstrates how modern **transformer based vision models** can be used for **terrain perception in robotics and exploration systems**.

---

# Demo

### Example Perception Output

```
Camera Feed | Segmentation Map | Terrain Perception Dashboard
```

Example pipeline:

```
Input Frame → AI Segmentation → Terrain Interpretation → Rover Dashboard
```

The final output simulates a **rover style perception interface** similar to systems used in autonomous exploration vehicles.

---

# Problem Statement

Autonomous systems are typically designed for structured environments such as roads and cities.

However many real-world applications require navigation through **unstructured terrain**, including:

• deserts
• forests
• agricultural environments
• remote exploration zones

Understanding terrain is essential for:

• obstacle detection
• path planning
• safe navigation

This project builds an AI perception system capable of **interpreting complex off-road terrain**.

---

# System Architecture

The system processes camera images through multiple stages.

```
Camera Frame
     │
     ▼
Image Preprocessing
     │
     ▼
Feature Extraction (DINOv2 Vision Transformer)
     │
     ▼
ConvNeXt Segmentation Head
     │
     ▼
Semantic Segmentation Map
     │
     ▼
Visualization Engine
     │
     ▼
Terrain Perception Dashboard
```

---

# Model Architecture

The segmentation model combines **transformer feature extraction** with a lightweight segmentation head.

## Backbone

**DINOv2 Vision Transformer (ViT-S/14)**

DINOv2 is a self supervised model capable of producing powerful visual representations without requiring large labeled datasets.

Advantages:

• strong visual embeddings
• generalizable features
• robust representation learning

---

## Segmentation Head

A ConvNeXt style segmentation head converts transformer tokens into pixel level predictions.

Architecture:

```
Patch Tokens
    │
    ▼
Conv2D Stem
    │
    ▼
Depthwise Convolution
    │
    ▼
Pointwise Convolution
    │
    ▼
Segmentation Classifier
    │
    ▼
Upsampled Segmentation Map
```

---

# Terrain Classes

The system detects the following terrain categories.

| ID | Terrain Class  |
| -- | -------------- |
| 0  | Background     |
| 1  | Trees          |
| 2  | Lush Bushes    |
| 3  | Dry Grass      |
| 4  | Dry Bushes     |
| 5  | Ground Clutter |
| 6  | Flowers        |
| 7  | Logs           |
| 8  | Rocks          |
| 9  | Landscape      |
| 10 | Sky            |

Each pixel in the image is classified into one of these terrain categories.

---

# Dataset

The model was trained on a simulated off-road segmentation dataset.

Dataset layout:

```
Offroad_Segmentation_Training_Dataset
│
├── train
│   ├── Color_Images
│   └── Segmentation
│
├── val
│   ├── Color_Images
│   └── Segmentation
│
Offroad_Segmentation_testImages
│
├── Color_Images
└── Segmentation
```

Dataset statistics:

Training images: **2857**
Validation images: **317**
Test images: **1002**

---

# Training Pipeline

### Image Resolution

```
480 × 270
```

### Data Augmentation

• random horizontal flip
• brightness and contrast jitter
• normalization

### Loss Function

```
CrossEntropy Loss
```

### Optimizer

```
AdamW
learning rate: 1e-4
```

### Training Configuration

Batch size: 2
Epochs: 20
GPU: NVIDIA RTX 3050

---

# Model Performance

| Metric         | Score |
| -------------- | ----- |
| Mean IoU       | 0.37  |
| Pixel Accuracy | 0.76  |

While the system demonstrates good pixel accuracy, segmentation performance can be further improved with advanced architectures.

---

# Visualization Pipeline

To simulate real robotics perception systems, the output is visualized as a **terrain perception dashboard**.

Each frame includes:

```
Camera View | Semantic Segmentation | AI Terrain Interpretation
```

Features:

• colored terrain segmentation
• labeled terrain regions
• bounding boxes for detected terrain
• rover style perception interface

Small fragmented segments are filtered to maintain clean visualization.

---

# Output Structure

```
predictions
│
├── masks
│   Raw predicted segmentation masks
│
├── masks_color
│   Colored segmentation maps
│
└── overlay
    Terrain perception dashboard frames
```

---

# Video Generation

After inference, frames can be converted into a demo video.

Example command:

```
ffmpeg -framerate 24 -pattern_type glob -i "predictions/overlay/*.png" -pix_fmt yuv420p terrain_perception_demo.mp4
```

---

# Applications

This system can support multiple real world domains.

• autonomous off-road vehicles
• agricultural robotics
• search and rescue robots
• exploration rovers
• autonomous drones

Terrain perception is critical for **navigation in environments without structured infrastructure**.

---

# Future Improvements

Potential improvements include:

• DeepLabV3 or SegFormer segmentation architectures
• fine tuning the transformer backbone
• class balancing techniques
• terrain traversability estimation
• deployment on embedded robotics platforms

---

# Repository Structure

```
project/
│
├── train_segmentation.py
├── test_segmentation.py
├── segmentation_head.pth
│
├── predictions/
│
├── dataset/
│
└── README.md
```

---

# Tech Stack

Python
PyTorch
Torchvision
OpenCV
DINOv2
FFmpeg

---

# Conclusion

This project demonstrates how modern vision transformers combined with semantic segmentation can enable **terrain understanding in unstructured environments**.

Such systems are essential for the next generation of **autonomous robotics and exploration platforms**.


