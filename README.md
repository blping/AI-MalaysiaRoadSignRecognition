# AI-MalaysiaRoadSignRecognition
A machine learning project developed for the Artificial Intelligence assignment to investigate the robustness of road sign recognition models under adverse Malaysian environmental conditions.

This project compares three machine learning approaches:

- **Convolutional Neural Network (CNN)**
- **Support Vector Machine (SVM)**
- **You Only Look Once (YOLO)**

The project uses synthetic data augmentation to simulate adverse conditions such as rain, haze, night glint, deformation, and occlusion.

---

## Features

- Malaysian road sign recognition
- Supports **10 road sign classes**
- Supports three machine learning approaches:
  - Convolutional Neural Network (CNN)
  - Support Vector Machine (SVM)
  - You Only Look Once (YOLO)
- Clean and augmented dataset evaluation
- Synthetic adverse condition simulation:
  - Tropical rain
  - Haze / low visibility
  - Night driving and light glint
  - Physical and perspective deformation
  - Occlusion and surface dirt
- Multiple training configurations:
  - 40% augmented
  - 50% augmented
  - 60% augmented
  - 100% augmented
- Two-phase training methodology:
  - Phase 1: Baseline training using clean data
  - Phase 2: Robust training using augmented data
- Unified image preprocessing
- Test-Time Augmentation (TTA)
- Model performance evaluation

---

## Road Sign Classes

The system supports the following road sign classes:

- Bumps
- No U-turn
- No Entry
- No Parking
- Obstruction
- Pass on the left
- Roadway diverges
- Speed limit
- Stop
- U-turn

---

## Built With

- Python
- OpenCV
- TensorFlow / Keras
- MobileNetV2
- Scikit-learn
- HOG (Histogram of Oriented Gradients)
- PCA (Principal Component Analysis)
- YOLOv8
- NumPy
- Matplotlib

---

## Models

### Convolutional Neural Network (CNN)

- Uses **MobileNetV2**
- Uses transfer learning
- Base model is initially frozen
- Higher-level layers are unfrozen during fine-tuning
- Trained using clean and augmented datasets

### Support Vector Machine (SVM)

- Uses **HOG-PCA** feature extraction
- Uses an **RBF kernel**
- Uses grid search and cross-validation for parameter tuning

### You Only Look Once (YOLO)

- Uses **YOLOv8**
- Performs road sign object detection and localization
- Uses pre-trained YOLOv8 weights
- Fine-tuned using the project dataset

---

## Dataset

The project uses a Malaysian road sign dataset containing **10 road sign classes**.

### Dataset A — Clean Dataset

Contains original road sign images captured under clear conditions.

Used for:

- Baseline training
- Clean test evaluation
- Comparing model performance under ideal conditions

### Dataset B — Augmented Dataset

Generated from the clean dataset using synthetic environmental and physical transformations.

The augmentation conditions include:

| Augmentation | Description |
|---|---|
| Tropical Rain | Simulates rainfall, noise, and motion blur |
| Haze | Simulates reduced visibility and atmospheric haze |
| Night / Glint | Simulates nighttime brightness reduction and light reflections |
| Deformation | Simulates perspective changes and physical distortion |
| Occlusion / Surface Dirt | Partially blocks road sign features |

---

## Preprocessing

A unified preprocessing pipeline is applied to the three models.

- Resize images to **224 × 224 pixels**
- Convert images to the required colour space
- Apply **CLAHE** to the luminance channel
- Apply **3 × 3 Gaussian blur**
- Perform model-specific feature extraction or normalization

For SVM:

- HOG feature extraction
- PCA dimensionality reduction
- RBF kernel classification

---

## Training Methodology

The project follows a **two-phase training strategy**.

### Phase 1 — Baseline Training

Models are trained using clean, non-augmented data to establish baseline performance under ideal conditions.

### Phase 2 — Robust Training

The Phase 1 models are further trained using augmented datasets with different clean-to-augmented data ratios.

Tested configurations include:

- 40% augmented
- 50% augmented
- 60% augmented
- 100% augmented

This allows the project to investigate the trade-off between clean-data accuracy and robustness under adverse conditions.

---

## Test-Time Augmentation

**Test-Time Augmentation (TTA)** is used during inference to improve prediction stability.

The system:

1. Generates multiple transformed versions of the same test image.
2. Passes the transformed images through the trained model.
3. Collects the resulting predictions and confidence scores.
4. Aggregates the results to produce a more reliable prediction.

---

## Model Performance

The models are evaluated using both:

- **Clean Test Data**
- **Augmented Test Data**

### CNN

| Test Dataset | Accuracy |
|---|---:|
| Clean Test Data | **90.79%** |
| Augmented Test Data | **57.16%** |

After robust training, the **60% clean / 40% augmented** configuration achieved:

- Clean Test Accuracy: **92.34%**
- Augmented Test Accuracy: **71.70%**

### SVM

| Test Dataset | Accuracy |
|---|---:|
| Clean Test Data | **86.46%** |
| Augmented Test Data | **74.80%** |

The fully augmented SVM achieved:

- Augmented Test Accuracy: **97.27%**
- Clean Test Accuracy: **83.79%**

### YOLO

| Test Dataset | mAP50 |
|---|---:|
| Clean Test Data | **94.06%** |
| Augmented Test Data | **68.38%** |

After robust training, the YOLO models showed improved performance on augmented test data.

## How to Run

### 1. Clone the Repository

```bash
git clone <repository-url>
cd <project-folder>
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare the Dataset

Place the dataset according to the required directory structure.

### 4. Run Preprocessing

Run the required preprocessing and augmentation scripts.

### 5. Train the Model

Select and train the required model:

- CNN
- SVM
- YOLO

### 6. Run Testing

Run the testing scripts using:

- Clean test dataset
- Augmented test dataset

### 7. Review Results

The generated results may include:

- Accuracy
- Precision
- Recall
- F1-score
- mAP50
- Confusion matrices
- Prediction samples

---

## Team Members

| Name | Contribution |
|---|---|
| Tong Zi Chen | Convolutional Neural Networks (CNN) |
| Go Chun Hong | Support Vector Machines (SVM) |
| Pang Qian Fu | You Only Look Once (YOLO) |

---

## Notes

- The adverse environmental conditions are **synthetically generated**.
- Synthetic augmentation may not fully represent real-world Malaysian weather conditions.
- The dataset contains a limited number of Malaysian road sign classes.
- Test-Time Augmentation introduces additional computational overhead during inference.
- The project focuses on evaluating **model robustness** rather than deploying a complete autonomous driving system.
- The current dataset contains a limited number of Malaysian road sign classes.
- Test-Time Augmentation introduces additional computational overhead during inference.
- The project focuses on evaluating model robustness rather than deploying a complete autonomous driving system. 
