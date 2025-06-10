# 🧠 Tampered Image Patch Classification with Explainability

## Overview

This project implements a **deep learning pipeline** using PyTorch to classify image patches as **original or tampered** from the CG-1050 dataset. It includes data preprocessing, model training (basic CNN and ResNet-18), evaluation using metrics and Grad-CAM visualizations, and result analysis through confusion matrices and heatmaps.

---

## 📁 Project Structure

| Notebook | Purpose |
|----------|---------|
| `DataSet_Preparation.ipynb` | Loads file paths, creates dataset labels, and performs train/val/test split |
| `Data_Preprocessing_Enhancements.ipynb` | Prepares PyTorch `Dataset` objects with image resizing, normalization, and augmentation |
| `CNN_Model_Basic_PyTorch.ipynb` | Implements and trains a custom convolutional neural network |
| `ResNet18_Improved_Model.ipynb` | Fine-tunes a pretrained ResNet-18 model for binary classification |
| `Full_Compiled_Script.ipynb` | Integrates dataset, model, training loop, and logging in one end-to-end pipeline. |
| `Evaluating_Trained_Model.ipynb` | Loads trained weights and evaluates the model using standard classification metrics |
| `ResNet18_Confusion_Matrix.ipynb` | Generates and visualizes confusion matrix using `sklearn.metrics` |
| `Grad-CAM_Tampered_Images.ipynb` | Applies Grad-CAM to tampered image predictions to visualize activation maps |
| `Grad-CAM_50_Random_Tampered_Images.ipynb` | Batch heatmaps for random tampered images |
| `Grad-CAM_Resnet18_HeatMap_Misclassified_Results.ipynb` | Visualizes attention for misclassified patches |

---

## 🧠 Code Structure

### 1. **Data Preparation & Preprocessing**
- **Loading**: Original and tampered image paths.
- **Labeling**: Binary classification (`0 = Original`, `1 = Tampered`).
- **Split**: Train, Validation, and Test sets.
- **Enhancements**: Image resizing (128×128), normalization, and optional augmentation.

### 2. **Model Building**
- **Basic CNN**: Built with sequential Conv2D, ReLU, Pooling, and Linear layers.
- **ResNet-18**: Imported from `torchvision.models`, with a modified final layer for binary output.

### 3. **Model Training & Evaluation**
- Full training loop with:
  - Accuracy and loss tracking
  - Validation checks
  - Model saving
- Evaluated using:
  - Accuracy
  - Precision, Recall, F1 Score

### 4. **Explainability with Grad-CAM**
- Heatmap visualization of model focus regions during classification.
- Helps interpret model predictions and localize manipulated areas.

### 5. **Performance Metrics**
- Visualized via confusion matrices.
- Highlights class-wise accuracy and error patterns.

---

## ⚙️ Key Technologies
- **Python**, **PyTorch**, **OpenCV**, **Matplotlib**, **Torchvision**, **Grad-CAM**
- **CNN and ResNet-18 architectures**
- Explainability via **Class Activation Maps (CAM)**

---

## 🧪 Results Summary
- ResNet-18 achieved higher classification accuracy than the basic CNN.
- Grad-CAM visualizations show reliable focus on tampered image regions.
- The model generalizes well to unseen patches and localizes forgery effectively.
