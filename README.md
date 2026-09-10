# 🚗 Driver Drowsiness Detection System

A **real-time driver drowsiness detection system** built using **YOLOv5, PyTorch, and OpenCV** to detect whether a driver is **awake or experiencing drowsiness** from a camera/video feed.

The primary objective of this project is to provide a **cost-effective, camera-based approach to road-safety monitoring** without requiring specialized infrared or physiological sensors.

---

## 📌 Project Overview

Drowsy driving is a major road-safety concern. Traditional driver-monitoring systems may rely on expensive infrared cameras or intrusive physiological sensors.

This project uses **computer vision and deep learning** to analyze a driver's face through a camera feed and classify the driver's state into two categories:

* 🟢 **Awake**
* 🔴 **Drowsiness**

The system uses **YOLOv5s (You Only Look Once v5 - Small)** with **transfer learning** from COCO-pretrained weights and fine-tunes the model on a custom driver-drowsiness dataset.

---

## ✨ Key Features

* 🎥 Real-time webcam/video detection
* 🧠 YOLOv5s-based object detection
* 😴 Drowsiness detection
* 👁️ Awake/alert state detection
* 📦 Bounding-box based face detection
* ⚡ Lightweight YOLOv5s architecture
* 🔄 Transfer learning using COCO-pretrained weights
* 📊 Validation using Precision, Recall and mAP
* 🚨 Warning mechanism when drowsiness is detected
* 📈 Training and evaluation visualization

---

## 🛠️ Tech Stack

| Category                | Technology          |
| ----------------------- | ------------------- |
| Programming Language    | Python 3.x          |
| Deep Learning Framework | PyTorch             |
| Object Detection        | YOLOv5s             |
| Computer Vision         | OpenCV              |
| Image Processing        | Pillow              |
| Data Processing         | NumPy, Pandas       |
| Visualization           | Matplotlib, Seaborn |
| Experimentation         | Jupyter Notebook    |
| Experiment Logging      | TensorBoard         |
| Configuration           | PyYAML              |
| Scientific Computing    | SciPy               |
| Performance Utilities   | tqdm, psutil, thop  |

---

## 🧠 Model Architecture

The project uses **YOLOv5s**, the small and lightweight variant of YOLOv5.

The model uses a single-stage object detection approach:

```text
Input Image / Video Frame
          ↓
     Preprocessing
     320 × 320
          ↓
     YOLOv5s Backbone
          ↓
   Feature Extraction
          ↓
     Detection Head
          ↓
 Bounding Box + Class
          ↓
 ┌─────────────────────┐
 │                     │
Awake              Drowsiness
 │                     │
 ↓                     ↓
No Alert          Warning Trigger
```

The model was initialized using **COCO-pretrained `yolov5s.pt` weights** and fine-tuned on the driver-drowsiness dataset.

The final detection head was configured for **2 classes**.

---

## 📂 Dataset

The dataset contains labeled images of drivers/faces categorized into:

1. **Awake**
2. **Drowsiness**

The project uses the **YOLO annotation format**, where bounding boxes are provided for the detected driver/face.

### Dataset Structure

```text
Dataset/
├── images/
│   ├── train/
│   └── val/
│
└── labels/
    ├── train/
    └── val/
```

Training images were resized to:

```text
320 × 320 pixels
```

Dataset configuration is provided through:

```text
dataset.yml
```

---

## 🔄 Project Workflow

```text
Camera / Video Input
        ↓
Frame Capture using OpenCV
        ↓
Image Preprocessing
        ↓
Resize to 320 × 320
        ↓
YOLOv5s Inference
        ↓
Bounding Box + Class Prediction
        ↓
     Classification
      ↙         ↘
   Awake      Drowsiness
     ↓            ↓
 No Alert    Warning Trigger
```

### Step-by-Step

1. **Data Collection & Annotation**

   * Driver images are collected for awake and drowsy states.
   * Images are annotated using YOLO bounding boxes.

2. **Dataset Preparation**

   * Images and labels are separated into training and validation sets.
   * Dataset configuration is specified in `dataset.yml`.

3. **Transfer Learning**

   * COCO-pretrained `yolov5s.pt` weights are loaded.

4. **Model Training**

   * YOLOv5 training is performed using `train.py`.

5. **Validation**

   * The trained model is evaluated using `val.py`.
   * Precision, Recall and mAP are calculated.

6. **Inference**

   * `detect.py` is used for detection on webcam/video input.

7. **Experimentation**

   * Multiple experiments were conducted for iterative model development and hyperparameter tuning.

---

## ⚙️ Training Configuration

The final training experiment (`exp5`) used:

| Parameter               |        Value |
| ----------------------- | -----------: |
| Base Model              |      YOLOv5s |
| Epochs                  |          500 |
| Batch Size              |           16 |
| Image Size              |    320 × 320 |
| Optimizer               |          SGD |
| Initial Learning Rate   |         0.01 |
| Final Learning Rate     |         0.01 |
| Momentum                |        0.937 |
| Weight Decay            |       0.0005 |
| Warmup Epochs           |            3 |
| Early Stopping Patience |          100 |
| Workers                 |            2 |
| Mosaic Augmentation     |      Enabled |
| Horizontal Flip         |          0.5 |
| HSV Augmentation        |      Enabled |
| Scale Augmentation      |          0.5 |
| LR Scheduler            | Linear Decay |
| IoU Threshold           |          0.2 |

---

## 📊 Model Performance

The final experiment (`exp5`) reported:

| Metric                         |             Result |
| ------------------------------ | -----------------: |
| **mAP@0.5**                    |         **~99.5%** |
| **mAP@0.5:0.95**               | **~0.847 – 0.851** |
| **Precision**                  |  **~99.4 – 99.9%** |
| **Recall**                     |   **~99.8 – 100%** |
| Validation Box Loss            |            ~0.0118 |
| Validation Objectness Loss     |            ~0.0032 |
| Validation Classification Loss |            ~0.0013 |

The model reached approximately **99.5% mAP@0.5** and stabilized from around epoch 220 onward according to the project experiments.

> **Note:** These metrics are based on the project's reported validation results. Real-world performance can vary depending on lighting, camera position, driver appearance, occlusion, and environmental conditions.

---

## 📈 Training Results

The final training experiment generated:

```text
runs/train/exp5/
│
├── results.csv
├── results.png
├── F1_curve.png
├── PR_curve.png
├── P_curve.png
├── R_curve.png
├── confusion_matrix.png
├── labels.jpg
│
├── train_batch*.jpg
├── val_batch*_pred.jpg
│
├── weights/
│   ├── best.pt
│   └── last.pt
│
└── events.out.tfevents.*
```

These artifacts can be used to analyze:

* Training and validation losses
* Precision
* Recall
* F1 score
* Precision-Recall relationship
* Confusion matrix
* Label distribution
* Validation predictions
* Model weights

---

## 📁 Project Structure

```text
yolo/
│
├── Drowsiness_detect_YOLO (1).ipynb
│
├── Recording 2025-01-20 161933.mp4
├── WIN_20250120_*.mp4
├── images.jpeg
├── images.avif
│
└── yolov5-master/
    │
    ├── train.py
    ├── detect.py
    ├── val.py
    ├── export.py
    ├── benchmarks.py
    │
    ├── dataset.yml
    ├── yolov5s.pt
    ├── requirements.txt
    │
    ├── models/
    │   ├── yolo.py
    │   ├── common.py
    │   └── yolov5s.yaml
    │
    ├── utils/
    │   ├── dataloaders.py
    │   ├── general.py
    │   ├── augmentations.py
    │   ├── loss.py
    │   ├── metrics.py
    │   ├── plots.py
    │   └── flask_rest_api/
    │
    ├── data/
    ├── segment/
    ├── classify/
    │
    └── runs/
        └── train/
            ├── exp/
            ├── exp2/
            ├── exp3/
            ├── exp4/
            └── exp5/
```

---

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd <your-project-folder>
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate on Windows:

```bash
venv\Scripts\activate
```

On Linux/macOS:

```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🏋️ Training the Model

The YOLOv5 model can be trained using:

```bash
python train.py
```

The training process uses the dataset configuration specified in:

```text
dataset.yml
```

The final experiment was trained for **500 epochs** using YOLOv5s transfer learning.

---

## 🔍 Running Detection

For inference on a webcam or video source:

```bash
python detect.py
```

The detection pipeline captures frames, preprocesses them, runs YOLOv5 inference, and produces bounding boxes with the predicted class.

For a drowsiness prediction, the system triggers the warning mechanism.

---

## 📓 Jupyter Notebook

The project includes an interactive notebook:

```text
Drowsiness_detect_YOLO (1).ipynb
```

The notebook was used for experimentation, development and testing of the drowsiness detection pipeline.

---

## 🧪 Experiments

Multiple training experiments were performed:

```text
exp
exp2
exp3
exp4
exp5
```

These experiments were used for iterative model development and hyperparameter tuning.

`exp5` represents the final/best reported training experiment.

---

## 📦 Model Weights

The trained model weights are stored under:

```text
runs/train/exp5/weights/
```

Important files:

```text
best.pt
last.pt
```

`best.pt` represents the best-performing saved model during training.

---

## 🌐 Deployment Potential

The YOLOv5 ecosystem supports exporting the trained model to multiple formats:

* **ONNX** — cross-platform deployment
* **TensorFlow Lite** — mobile and embedded deployment
* **TensorRT** — NVIDIA GPU-optimized inference
* **CoreML** — Apple devices
* **OpenVINO** — Intel edge devices
* **Flask REST API** — web-based deployment

This provides a foundation for future **edge, embedded, mobile or web-based driver-monitoring applications**.

---

## 💡 Key Strengths

### Lightweight Model

YOLOv5s is a relatively compact model suitable for real-time inference.

### Transfer Learning

COCO-pretrained weights were used to initialize the model before fine-tuning on the drowsiness dataset.

### Real-Time Detection

The system supports inference on live webcam/video input.

### Strong Validation Results

The final experiment reported:

```text
mAP@0.5     ≈ 99.5%
Precision   ≈ 99.4–99.9%
Recall      ≈ 99.8–100%
```

### Multiple Deployment Options

The trained model can be exported into several formats for future deployment.

---

## 🔮 Future Improvements

Possible future improvements include:

* Integration with an audible alarm system
* Integration with vehicle hardware
* Mobile/edge-device deployment
* TensorFlow Lite deployment
* ONNX/TensorRT optimization
* Larger and more diverse datasets
* Testing under different lighting conditions
* Night-time driver monitoring
* Improved robustness against occlusion
* Integration with vehicle sensors
* Web-based monitoring using the Flask REST API

---

## ⚠️ Limitations

The reported performance is based on the available validation dataset. Real-world deployment may introduce challenges such as:

* Poor lighting
* Night-time driving
* Different camera angles
* Face occlusion
* Sunglasses or other obstructions
* Different driver appearances
* Camera quality
* Environmental variation

Therefore, the reported validation metrics should not be interpreted as a guarantee of equivalent performance in every real-world driving condition.

---

## 👩‍💻 Project Highlights

* Implemented a **computer-vision-based driver monitoring system**
* Used **YOLOv5s transfer learning**
* Built a **two-class drowsiness detection model**
* Prepared and trained on **YOLO-format annotated data**
* Performed **500-epoch training**
* Evaluated the model using **mAP, Precision and Recall**
* Conducted multiple training experiments
* Tested the system using **webcam/video input**
* Generated detailed training and evaluation visualizations

---

## 📚 Learning Outcomes

Through this project, the following concepts were explored:

* Object detection
* YOLO architecture
* Transfer learning
* Deep learning model training
* Bounding-box annotations
* Dataset preparation
* Data augmentation
* Model validation
* Precision, Recall and mAP
* Real-time computer vision
* PyTorch
* OpenCV
* Model export and deployment

---

## 📄 Project Summary

This project demonstrates an end-to-end **driver drowsiness detection pipeline** using **YOLOv5s transfer learning**, covering dataset preparation, model training, validation and real-time inference.

The final experiment reported approximately **99.5% mAP@0.5**, with Precision and Recall above 99% on the validation data, while the lightweight YOLOv5s architecture provides a foundation for future real-time and edge-device deployment.

---

## ⭐ Acknowledgement

This project was developed as a practical application of **deep learning and computer vision** for road-safety monitoring, with a focus on real-time driver drowsiness detection.
