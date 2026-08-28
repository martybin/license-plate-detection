<div align="center">

# 🇮🇷 Iranian License Plate Recognition
### Mine Gate • Transfer Learning Edition

**Real-Time Iranian License Plate Detection, Recognition & Vehicle Verification**

<br/>

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Detection-orange?style=for-the-badge)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c?style=for-the-badge&logo=pytorch)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5c3ee8?style=for-the-badge&logo=opencv)
![SQLite](https://img.shields.io/badge/SQLite-Database-003b57?style=for-the-badge&logo=sqlite)

<br/>

*A production-oriented computer vision pipeline for recognizing Iranian license plates at mining truck entry gates.*

</div>

---

## 📌 Overview

**Iranian License Plate Recognition (Mine Gate)** is an end-to-end real-time LPR system designed specifically for **mining environments**.

The system detects an Iranian license plate from a camera stream, preprocesses the detected plate, recognizes its characters, validates the Iranian plate format, and finally looks up the recognized plate in a local SQLite database to retrieve vehicle and driver information.

The project uses **transfer learning at both major deep-learning stages**:

```text
Camera
   │
   ▼
YOLOv8
Plate Detection
   │
   ▼
Image Preprocessing
   │
   ▼
ResNet (ImageNet)
   │
   ▼
BiLSTM
   │
   ▼
CTC
Character Recognition
   │
   ▼
Iranian Plate Validation
   │
   ▼
SQLite Vehicle Database
   │
   ▼
Mine Gate Monitor
```

The system is intended to handle the challenging conditions commonly found around mines, including:

- Dust-covered plates
- Tilted plates
- Low-contrast plates
- Slightly blurred images
- Variable lighting conditions
- Real-time camera streams

---

# ✨ Features

- 🎯 **YOLOv8** license plate detection
- 🧠 **Transfer learning from COCO** for plate detection
- 🔤 **ResNet + BiLSTM + CTC** OCR architecture
- 🧠 **ImageNet-pretrained ResNet** backbone
- 🇮🇷 Iranian license plate format validation
- 🧹 Advanced image preprocessing
- 📹 Webcam support for development/testing
- 📡 RTSP support for industrial cameras
- 🗃️ SQLite vehicle and driver database
- 🟢 Allowed vehicle detection
- 🔴 Denied / unknown vehicle detection
- 🖥️ Fullscreen mine-gate monitoring
- 🔄 Stable consecutive readings to reduce recognition flickering
- 🧩 Modular architecture
- 🚀 Separate detector and recognizer training pipelines

---

# 🧠 Transfer Learning

Transfer learning is one of the main design decisions of this project.

Instead of training every component from scratch, the system starts from models that have already learned useful visual representations from large datasets.

## 🎯 YOLOv8 — COCO Pretraining

The plate detector uses **YOLOv8**, initialized from COCO-pretrained weights and fine-tuned for license plate detection.

This provides a strong starting point for learning the visual characteristics of license plates.

## 🧠 ResNet — ImageNet Pretraining

The OCR recognizer uses an **ImageNet-pretrained ResNet backbone**.

The pretrained network already contains useful visual representations for:

- Edges
- Textures
- Shapes
- Character structures
- Local visual patterns

These features are then adapted to the Iranian license plate recognition task.

## 🚀 Why Transfer Learning?

Compared with training a CRNN completely from scratch, transfer learning provides:

- Better generalization
- Faster convergence
- Less training data required
- Stronger initial visual representations
- Better performance on degraded real-world images

> **Recommendation:** Keep `pretrained: true` in `configs/config.yaml`.

---

# 🏗️ System Architecture

```text
┌──────────────────────────────────────────────┐
│              📷 Camera Source                │
│          Webcam / Industrial RTSP            │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│                 🎯 YOLOv8                    │
│            License Plate Detection           │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│            🧹 Image Processing                │
│                                              │
│  Perspective Correction                     │
│  Deskew                                      │
│  Bilateral Filtering                         │
│  CLAHE                                       │
│  Sharpening                                  │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│        🧠 ResNet — ImageNet Pretrained       │
│             Feature Extraction               │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│                  BiLSTM                      │
│           Sequence Modeling                  │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│                   CTC                        │
│            Character Decoding                │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│          🇮🇷 Plate Validation & Formatting    │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│              🗃️ SQLite Database              │
│          Vehicle / Driver Lookup             │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│             🖥️ Gate Monitor                  │
│       ALLOWED / DENIED / UNKNOWN             │
└──────────────────────────────────────────────┘
```

---

# 📁 Project Structure

```text
iranian_lpr/
│
├── configs/
│   └── config.yaml
│
├── models/
│   ├── __init__.py
│   ├── detector.py
│   ├── recognizer.py
│   └── pipeline.py
│
├── utils/
│   ├── __init__.py
│   ├── image_processing.py
│   ├── plate_utils.py
│   └── database.py
│
├── inference/
│   ├── __init__.py
│   └── realtime.py
│
├── training/
│   ├── __init__.py
│   ├── train_detector.py
│   ├── train_recognizer.py
│   └── prepare_dataset.py
│
├── weights/
│
├── data/
│
├── main.py
├── requirements.txt
└── README.md
```

## 📂 Directory Responsibilities

| Directory / File | Responsibility |
|---|---|
| `configs/` | Central system configuration |
| `models/` | Detection, recognition and complete inference pipeline |
| `utils/` | Image processing, plate validation and database utilities |
| `inference/` | Real-time camera inference |
| `training/` | Dataset preparation and model training |
| `weights/` | Trained model weights |
| `data/` | Datasets and SQLite database |
| `main.py` | Application entry point |
| `requirements.txt` | Python dependencies |
| `README.md` | Project documentation |

---

# ⚙️ 1. Environment Setup

Create a virtual environment:

```bash
python -m venv venv
```

### Linux / macOS

```bash
source venv/bin/activate
```

### Windows

```bash
venv\Scripts\activate
```

Install the project dependencies:

```bash
pip install -r requirements.txt
```

---

## 🖥️ Check CUDA

To verify whether PyTorch can access CUDA:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

If the output is:

```text
True
```

CUDA is available to PyTorch.

> ⚠️ The system can run on CPU, but **training will be very slow**. An NVIDIA GPU with CUDA support is strongly recommended.

---

# 📦 2. Dataset Preparation

The input dataset is expected to contain three splits:

```text
train/
validation/
test/
```

Each split contains:

```text
.jpg images
.xml annotation files
```

The XML files contain **character-level annotations**.

---

## 📝 Step 2.1 — Configure Dataset Paths

Open:

```text
training/prepare_dataset.py
```

Configure the actual dataset locations:

```python
TRAIN_FOLDER = Path(r"/path/to/your/train")
VAL_FOLDER   = Path(r"/path/to/your/validation")
TEST_FOLDER  = Path(r"/path/to/your/test")      # or None
```

---

## 🚀 Step 2.2 — Prepare the Dataset

Run:

```bash
python -m training.prepare_dataset
```

The preparation script automatically:

1. Reads every XML annotation
2. Extracts the annotated characters
3. Sorts characters from left to right using `xmin`
4. Reconstructs the complete plate text
5. Renames/copies images for the OCR dataset
6. Creates YOLO-format labels
7. Creates a full-image bounding box for the plate detector
8. Builds the required dataset directory structure
9. Generates `data.yaml`

For example, a reconstructed plate may look like:

```text
32ی45955
```

---

## 📊 Example Preparation Output

```text
YOLO dataset ready → data/plate_dataset
  Train: 19381 images
  Val  : 8364 images

OCR dataset ready → data/ocr_dataset
Total success: 27397 | Total failed: 348
```

The exact numbers depend on the dataset and annotation quality.

---

# 🏋️ 3. Training

The project contains two independently trainable deep-learning components:

```text
┌─────────────────────────┐
│ YOLOv8                  │
│ Plate Detection         │
└─────────────────────────┘

┌─────────────────────────┐
│ ResNet + BiLSTM + CTC   │
│ Character Recognition   │
└─────────────────────────┘
```

They are trained separately.

---

# 🎯 3.1 Train the Detector

Run:

```bash
python -m training.train_detector
```

The detector uses YOLOv8 and is fine-tuned for Iranian license plate detection.

After training, the best weights are automatically copied to:

```text
weights/yolov8_plate.pt
```

### ⏱️ Training Time

Training approximately 20,000 images on CPU can take **several days**.

For practical training times, use an NVIDIA GPU.

---

# 🔤 3.2 Train the Recognizer

Run:

```bash
python -m training.train_recognizer
```

The OCR architecture is:

```text
Image
  │
  ▼
ResNet
  │
  ▼
BiLSTM
  │
  ▼
CTC
  │
  ▼
Character Sequence
```

The trained recognizer weights are saved to:

```text
weights/resnet_crnn_iran.pt
```

---

# 🧠 Recognizer Backbones

The supported ResNet backbones are configured in:

```text
configs/config.yaml
```

| Backbone | Speed | Accuracy | Recommended For |
|---|---|---|---|
| `resnet18` | ⚡ Fastest | Good | Real-time inference |
| `resnet34` | ⚖️ Balanced | Better | General deployment |
| `resnet50` | 🧠 Heaviest | Highest | Maximum accuracy |

Recommended starting point:

```yaml
recognizer:
  backbone: resnet18
  pretrained: true
```

> **Always keep `pretrained: true`** unless you intentionally want to train the backbone from scratch.

---

# 🗃️ 4. Vehicle & Driver Database

The system **does not identify people directly from their faces or images**.

Instead, it:

```text
License Plate
      ↓
OCR Text
      ↓
SQLite Lookup
      ↓
Vehicle / Driver Record
```

The database is automatically created at:

```text
data/vehicles.db
```

---

## ➕ Add a Vehicle

Example:

```python
from utils.database import VehicleDB

db = VehicleDB("data/vehicles.db")

db.upsert(
    plate="15م59754",
    driver_name="Ali Rezaei",
    national_id="0011223344",
    truck_id="TRK-087",
    vehicle_model="Volvo FH16",
    company="Sangan Mine",
    allowed=1,
    note="Main driver"
)
```

---

## 🧾 Database Schema

| Field | Description |
|---|---|
| `plate` | Exact plate text — primary key |
| `driver_name` | Driver full name |
| `national_id` | National ID number |
| `truck_id` | Internal truck ID |
| `vehicle_model` | Vehicle model |
| `company` | Company / mine name |
| `allowed` | `1` = allowed, `0` = denied |
| `note` | Optional free-text note |

---

# ⚙️ 5. Configuration

Main configuration file:

```text
configs/config.yaml
```

Example:

```yaml
camera:
  source: 0
  # source: "rtsp://user:pass@192.168.1.100:554/stream1"

recognizer:
  backbone: resnet18
  pretrained: true

detector:
  conf_threshold: 0.35
```

---

## 📹 Camera Configuration

### Webcam

For local testing:

```yaml
camera:
  source: 0
```

### Industrial RTSP Camera

For mine-gate deployment:

```yaml
camera:
  source: "rtsp://user:pass@192.168.1.100:554/stream1"
```

The application architecture allows the camera source to be changed without modifying the main inference pipeline.

---

# 🖥️ 6. Real-Time Inference

Once the models and configuration are ready:

```bash
python main.py
```

---

## 💻 Local Webcam Test

Configure:

```yaml
camera:
  source: 0
```

Then run:

```bash
python main.py
```

### Controls

| Key | Action |
|---|---|
| `q` | Quit |
| `Esc` | Quit |

### Detection Status

```text
🟢 Green → Allowed vehicle
🔴 Red   → Denied / Unknown vehicle
```

The recognized plate and driver information are displayed after **3 consecutive stable readings**.

This mechanism helps prevent UI flickering caused by occasional incorrect OCR predictions.

---

# ⛏️ 7. Mine Gate Deployment

For production deployment:

### 1️⃣ Connect the Hardware

Connect:

- Industrial camera
- Gate computer
- Guard/operator monitor

### 2️⃣ Configure the RTSP Stream

Edit:

```text
configs/config.yaml
```

Set:

```yaml
camera:
  source: "rtsp://user:pass@192.168.1.100:554/stream1"
```

### 3️⃣ Start the Application

```bash
python main.py
```

The application opens in **fullscreen mode** on the gate monitor.

---

## 👮 Example Gate Display

```text
Plate: 15 م 597 | 54   (0.91)
Driver : Ali Rezaei
National ID : 0011223344
Truck ID : TRK-087
Model : Volvo FH16
Company : Sangan Mine
Status : ALLOWED
```

This gives the gate operator an immediate view of:

- Recognized plate
- Recognition confidence
- Driver
- National ID
- Truck ID
- Vehicle model
- Company / mine
- Access status

---

# 🔄 8. Complete Runtime Workflow

The complete inference pipeline is:

```text
┌───────────────────┐
│ 1. Camera Frame   │
└─────────┬─────────┘
          ▼
┌───────────────────┐
│ 2. YOLOv8         │
│ Plate Detection   │
└─────────┬─────────┘
          ▼
┌───────────────────┐
│ 3. Plate Crop     │
└─────────┬─────────┘
          ▼
┌───────────────────┐
│ 4. Perspective    │
│    Correction     │
└─────────┬─────────┘
          ▼
┌───────────────────┐
│ 5. Deskew         │
└─────────┬─────────┘
          ▼
┌───────────────────┐
│ 6. Bilateral      │
│    Filter         │
└─────────┬─────────┘
          ▼
┌───────────────────┐
│ 7. CLAHE          │
└─────────┬─────────┘
          ▼
┌───────────────────┐
│ 8. Sharpening     │
└─────────┬─────────┘
          ▼
┌───────────────────┐
│ 9. ResNet         │
│ Feature Extraction│
└─────────┬─────────┘
          ▼
┌───────────────────┐
│ 10. BiLSTM        │
│ Sequence Modeling │
└─────────┬─────────┘
          ▼
┌───────────────────┐
│ 11. CTC           │
│ Character Decode  │
└─────────┬─────────┘
          ▼
┌───────────────────┐
│ 12. Iranian Plate │
│ Validation        │
└─────────┬─────────┘
          ▼
┌───────────────────┐
│ 13. Plate         │
│ Formatting        │
└─────────┬─────────┘
          ▼
┌───────────────────┐
│ 14. SQLite Lookup │
└─────────┬─────────┘
          ▼
┌───────────────────┐
│ 15. Stable Reading│
│ Verification      │
└─────────┬─────────┘
          ▼
┌───────────────────┐
│ 16. Gate Monitor  │
└───────────────────┘
```

### Step-by-step

1. Capture a frame from the camera.
2. YOLOv8 detects the license plate.
3. Crop the detected plate.
4. Correct perspective distortion.
5. Deskew the plate.
6. Apply bilateral filtering.
7. Improve local contrast using CLAHE.
8. Sharpen character boundaries.
9. Extract visual features using ImageNet-pretrained ResNet.
10. Model character sequences with BiLSTM.
11. Decode the sequence using CTC.
12. Validate the result against the Iranian plate format.
13. Format the recognized plate.
14. Search the plate in `data/vehicles.db`.
15. Verify consecutive stable readings.
16. Display the final result on the gate monitor.

---

# 🇮🇷 Iranian Plate Validation

The OCR output is passed through:

```text
utils/plate_utils.py
```

This component handles:

- Iranian plate validation
- Character/format checking
- Plate formatting
- Validation of the recognized sequence

The expected structure is represented as:

```text
XX L YYY ZZ
```

Where:

```text
XX  → Numeric section
L   → Iranian plate letter
YYY → Numeric section
ZZ  → Regional code
```

> Make sure the character set and formatting rules in `configs/config.yaml` and `utils/plate_utils.py` match the actual dataset and deployment requirements.

---

# 🚀 9. Performance Recommendations

## ⚡ Recommended Real-Time Configuration

Start with:

```text
Detector   → YOLOv8s
Recognizer → ResNet18 + BiLSTM + CTC
Pretrained → true
```

This provides a good balance between speed and accuracy.

---

## 🏎️ If Inference Speed Is the Priority

Use:

```text
resnet18
```

and a lightweight YOLOv8 configuration.

---

## 🎯 If Recognition Accuracy Is the Priority

Consider:

```text
resnet34
```

or:

```text
resnet50
```

These are particularly useful for heavily degraded plates.

---

## 📸 Improve the Dataset

For better production robustness, collect difficult samples such as:

- Dust-covered plates
- Tilted plates
- Night-time plates
- Low-light images
- Slightly blurred plates
- Low-contrast plates
- Different truck models
- Different camera distances
- Different camera angles
- Different weather conditions
- Different plate orientations

> Hard samples from the actual mining environment are especially valuable for improving production performance.

---

# 🧪 10. Troubleshooting

| Problem | Solution |
|---|---|
| No images found during training | Run `python -m training.prepare_dataset` again |
| Low recognition accuracy | Train longer, try `resnet34`/`resnet50`, and add more hard samples |
| Slow inference | Use `resnet18` and a smaller input size |
| Camera does not open | Check RTSP URL, credentials, network and camera availability |
| Invalid plate format | Check the character set and `plate_utils.py` |
| Weights are not saved | Ensure `weights/` exists and is writable |
| CPU training is extremely slow | Use an NVIDIA GPU with CUDA-enabled PyTorch |
| OCR result flickers | Verify camera quality and allow the stable-reading mechanism to work |
| Poor results on dusty plates | Add more dusty samples to the training dataset |
| Poor results at night | Add night-time and low-light samples |

---

# 🏭 11. Production Deployment

For production deployment at a mining gate, the following setup is recommended:

```text
                    ┌─────────────────┐
                    │ Industrial      │
                    │ Camera          │
                    └────────┬────────┘
                             │ RTSP
                             ▼
                    ┌─────────────────┐
                    │ Gate Computer   │
                    │                 │
                    │ Iranian LPR     │
                    │ Application     │
                    └────────┬────────┘
                             │
                ┌────────────┴────────────┐
                ▼                         ▼
       ┌─────────────────┐       ┌─────────────────┐
       │ SQLite Database │       │ Guard Monitor   │
       │ vehicles.db     │       │ Fullscreen UI   │
       └─────────────────┘       └─────────────────┘
```

---

## 🔄 Automatic Restart

Run the application under:

- `systemd`
- `supervisord`

This allows the service to restart automatically after:

- Application crashes
- System restarts
- Unexpected failures

---

## 📡 Fixed Industrial Camera

Point:

```yaml
camera:
  source: "rtsp://..."
```

to the permanent industrial camera stream.

The camera should ideally have:

- Stable positioning
- Consistent viewing angle
- Sufficient resolution
- Adequate lighting
- Reliable network connectivity

---

## 🗃️ Database Synchronization

Keep:

```text
data/vehicles.db
```

synchronized with the central vehicle management system when required.

The database contains the information required to determine whether a recognized vehicle is allowed through the gate.

---

## 🧩 Modular Model Replacement

The architecture is modular.

The detector and recognizer can be updated independently.

For example:

```text
New detector weights
        ↓
models/detector.py
        ↓
Existing OCR pipeline
```

or:

```text
Existing detector
        ↓
New recognizer weights
        ↓
models/recognizer.py
```

This means that a new detector or recognizer can be trained and deployed without rewriting the entire application.

---

# 📋 12. Quick Start

If the project is already configured, follow these steps:

### 1. Create environment

```bash
python -m venv venv
```

### 2. Activate environment

Linux/macOS:

```bash
source venv/bin/activate
```

Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure dataset paths

Edit:

```text
training/prepare_dataset.py
```

### 5. Prepare dataset

```bash
python -m training.prepare_dataset
```

### 6. Train detector

```bash
python -m training.train_detector
```

### 7. Train recognizer

```bash
python -m training.train_recognizer
```

### 8. Configure camera

Edit:

```text
configs/config.yaml
```

### 9. Populate database

Add the required vehicle records to:

```text
data/vehicles.db
```

### 10. Start the application

```bash
python main.py
```

---

# 📌 13. Important Files

| File | Purpose |
|---|---|
| `main.py` | Main application entry point |
| `configs/config.yaml` | Central system configuration |
| `models/detector.py` | YOLOv8 plate detector |
| `models/recognizer.py` | ResNet + BiLSTM + CTC OCR |
| `models/pipeline.py` | End-to-end processing pipeline |
| `models/__init__.py` | Python package initialization |
| `utils/image_processing.py` | Deskew, CLAHE, bilateral filter and sharpening |
| `utils/plate_utils.py` | Iranian plate validation and formatting |
| `utils/database.py` | SQLite vehicle/driver database |
| `utils/__init__.py` | Python package initialization |
| `inference/realtime.py` | Real-time camera loop and fullscreen overlay |
| `inference/__init__.py` | Python package initialization |
| `training/prepare_dataset.py` | XML/image dataset preparation |
| `training/train_detector.py` | YOLOv8 training |
| `training/train_recognizer.py` | OCR model training |
| `training/__init__.py` | Python package initialization |
| `weights/` | Trained model weights |
| `data/` | Dataset and database storage |
| `requirements.txt` | Python dependencies |
| `README.md` | Project documentation |

---

# 🛡️ 14. Production Checklist

Before deploying the system at the mine gate:

### Environment

- [ ] Python environment created
- [ ] Dependencies installed
- [ ] PyTorch installed correctly
- [ ] CUDA tested if using NVIDIA GPU

### Dataset

- [ ] Train folder configured
- [ ] Validation folder configured
- [ ] Test folder configured if available
- [ ] XML annotations available
- [ ] Dataset preparation completed
- [ ] YOLO dataset generated
- [ ] OCR dataset generated

### Models

- [ ] YOLOv8 detector trained
- [ ] Detector weights available
- [ ] OCR recognizer trained
- [ ] Recognizer weights available
- [ ] Correct ResNet backbone configured
- [ ] `pretrained: true` used during training

### Camera

- [ ] Industrial camera connected
- [ ] RTSP stream tested
- [ ] Network connection stable
- [ ] Camera position verified
- [ ] Lighting conditions tested

### Database

- [ ] `data/vehicles.db` created
- [ ] Required vehicles added
- [ ] Driver information verified
- [ ] `allowed` status verified
- [ ] Database synchronization strategy defined if required

### Application

- [ ] `configs/config.yaml` configured
- [ ] `python main.py` tested
- [ ] Fullscreen monitor tested
- [ ] Allowed vehicle flow tested
- [ ] Denied vehicle flow tested
- [ ] Unknown vehicle flow tested
- [ ] Stable-reading behavior tested
- [ ] Automatic restart configured

### Real-World Testing

- [ ] Dusty plates tested
- [ ] Tilted plates tested
- [ ] Blurred plates tested
- [ ] Low-contrast plates tested
- [ ] Night-time plates tested
- [ ] Different truck models tested
- [ ] Different plate distances tested

---

# 🎯 15. End-to-End Example

```text
                 🚛 MINING TRUCK
                       │
                       ▼
               📷 INDUSTRIAL CAMERA
                       │
                       ▼
                  🎯 YOLOv8
                PLATE DETECTION
                       │
                       ▼
              🧹 IMAGE PROCESSING
                       │
                       ▼
          🧠 RESNET + BILSTM + CTC
                OCR RECOGNITION
                       │
                       ▼
               🇮🇷 PLATE VALIDATION
                       │
                       ▼
                 🗃️ DATABASE
                       │
              ┌────────┴────────┐
              ▼                 ▼
         🟢 ALLOWED       🔴 DENIED
              │                 │
              └────────┬────────┘
                       ▼
                🖥️ GATE MONITOR
```

---

# 🔐 16. System Responsibilities

The project separates the responsibilities of the individual components:

```text
YOLOv8
  └── Where is the license plate?

Image Processing
  └── How can the plate image be improved?

ResNet
  └── What visual features exist in the plate?

BiLSTM
  └── What is the character sequence?

CTC
  └── How should the sequence be decoded?

Plate Utils
  └── Is this a valid Iranian plate?

SQLite
  └── Which vehicle/driver belongs to this plate?

Realtime Pipeline
  └── How should the result be displayed at the gate?
```

This separation keeps the system maintainable and makes future improvements easier.

---

# 🔮 17. Future Improvements

The current architecture provides a strong foundation for further improvements.

Possible extensions include:

- Multi-frame temporal voting
- More advanced OCR decoding
- Confidence-based OCR filtering
- Automatic plate tracking
- Multi-camera support
- Centralized database synchronization
- Gate barrier integration
- Entry/exit logging
- Historical vehicle reports
- Night-specific preprocessing
- Automatic exposure/brightness handling
- Hard-example mining
- Additional data augmentation
- Model quantization for edge deployment
- GPU/CPU inference optimization

---

# 🏁 Conclusion

**Iranian License Plate Recognition — Mine Gate** provides a complete pipeline for real-time Iranian license plate recognition in mining environments.

The architecture combines:

```text
YOLOv8
      +
COCO Transfer Learning
      +
Image Processing
      +
ImageNet Transfer Learning
      +
ResNet
      +
BiLSTM
      +
CTC
      +
Iranian Plate Validation
      +
SQLite
      +
Real-Time Camera Processing
      +
Mine Gate Monitoring
```

After the dataset has been prepared, the two models have been trained, the vehicle database has been populated, and the industrial camera has been configured, the system can be launched with:

```bash
python main.py
```

The result is a modular mine-gate LPR system capable of:

```text
Detect → Enhance → Recognize → Validate → Lookup → Decide → Display
```

---

<div align="center">

### 🇮🇷 Built for Iranian Mining Environments

**Dust • Tilt • Blur • Low Contrast • Real-Time Monitoring**

<br/>

**YOLOv8 × ResNet × BiLSTM × CTC**

</div>
