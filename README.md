# 🧠 Tampered Image Patch Classification with Explainability

## Overview

This project addresses the binary classification problem of **tampered vs. original image patches** using deep learning models trained on the **CG-1050** dataset. The pipeline involves preprocessing, training multiple CNN architectures (including a fine-tuned ResNet-18), evaluating their performance, and explaining model decisions via **Grad-CAM visualizations**.

---

## 📁 Project Structure

| Notebook | Purpose |
|----------|---------|
| `DataSet_Preparation.ipynb` | Loads file paths, creates dataset labels, and performs train/val/test split |
| `Data_Preprocessing_Enhancements.ipynb` | Prepares PyTorch `Dataset` objects with image resizing, normalization, and augmentation |
| `CNN_Model_Basic_PyTorch.ipynb` | Implements and trains a custom convolutional neural network |
| `ResNet18_Improved_Model.ipynb` | Fine-tunes a pretrained ResNet-18 model for binary classification |
| `FORGERYDETECTIONPROJ.ipynb` | Integrates dataset, model, training loop, and logging in one end-to-end pipeline |
| `Evaluating_Trained_Model.ipynb` | Loads trained weights and evaluates the model using standard classification metrics |
| `ResNet18_Confusion_Matrix.ipynb` | Computes and plots the confusion matrix for model predictions |
| `Grad-CAM_Tampered_Images.ipynb` | Applies Grad-CAM to tampered image predictions to visualize activation maps |
| `Grad-CAM_50_Random_Tampered_Images.ipynb` | Batch Grad-CAM visualization for random tampered patches |
| `Grad-CAM_Resnet18_HeatMap_Misclassified_Results.ipynb` | Examines model attention on misclassified patches |

---

## 🧪 Technical Pipeline

### 1. **Dataset Construction**
- **Source**: CG-1050 Dataset (Training images categorized into `ORIGINAL` and `TAMPERED`)
- Files are loaded and assigned binary labels (`0 = Original`, `1 = Tampered`)
- Data split:
  - **Train**: 70%
  - **Validation**: 15%
  - **Test**: 15%

### 2. **Preprocessing**
- Input images resized to `128x128x3`
- Normalization: pixel values scaled between 0 and 1 or standardized
- Augmentations include:
  - Horizontal/vertical flipping
  - Random rotations
  - Brightness/contrast shifts
- Converted to PyTorch `TensorDataset` and loaded via `DataLoader`

### 3. **Model Architectures**

#### A. **Custom CNN**
Defined in `CNN_Model_Basic_PyTorch.ipynb`:
```python
nn.Sequential(
    nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
    ...
    nn.Linear(...), nn.Sigmoid()
)
```
- Lightweight model trained using `Adam` optimizer and `BCELoss`
- Fast convergence but lower generalization compared to ResNet

#### B. **ResNet-18 Fine-tuned**
Modified in `ResNet18_Improved_Model.ipynb`:
- Loaded via `torchvision.models.resnet18(pretrained=True)`
- Final FC layer replaced with `nn.Linear(512, 2)` for binary classification
- Optional freezing of initial layers during fine-tuning
- Trained using `CrossEntropyLoss`

### 4. **Training Loop**
Implemented in `FORGERYDETECTIONPROJ.ipynb`:
- Epoch-wise training with:
  - Loss tracking
  - Accuracy monitoring on validation set
  - Model checkpoint saving based on validation performance
- Optimization:
  - Optimizer: `Adam`
  - Scheduler: `StepLR` (optional)
  - Early stopping: Based on validation loss

### 5. **Evaluation Metrics**
Computed in `Evaluating_Trained_Model.ipynb`:
- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score**
- Predictions saved for external visualization

### 6. **Confusion Matrix**
Plotted in `ResNet18_Confusion_Matrix.ipynb` using `sklearn.metrics.confusion_matrix` and `seaborn.heatmap`

---

## 🔍 Explainability: Grad-CAM

Grad-CAM implementations in 3 notebooks:
- Computes **activation maps** from the last convolutional layer
- Highlights **discriminative regions** for decision-making
- Applied to:
  - Random samples (`Grad-CAM_50_Random_Tampered_Images.ipynb`)
  - Misclassified patches (`Grad-CAM_Resnet18_HeatMap_Misclassified_Results.ipynb`)
  - Tampered images (`Grad-CAM_Tampered_Images.ipynb`)

Technical note:
```python
grads = torch.autograd.grad(score, conv_output)[0]
weights = grads.mean(dim=[2, 3], keepdim=True)
cam = F.relu((weights * conv_output).sum(dim=1, keepdim=True))
```

---

## 📊 Key Results

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Basic CNN | ~83% | ~82% | ~84% | ~83% |
| ResNet-18 | **~91%** | **~90%** | **~93%** | **~91%** |

- ResNet-18 shows significant improvement due to better feature reuse and deeper representation
- Grad-CAM confirms that ResNet-18 attends to manipulated boundaries more precisely
- Misclassifications often involve subtle tampering or high visual similarity to originals

---

## ⚙️ Tech Stack

- **Language**: Python 3.8+
- **Libraries**: PyTorch, Torchvision, OpenCV, PIL, Scikit-learn, Matplotlib, NumPy
- **Hardware Used**: GPU-enabled training (recommended for ResNet)
