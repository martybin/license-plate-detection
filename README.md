### Iranian License Plate Recognition (Mine Gate) – Transfer Learning Edition

High-accuracy real-time Iranian license plate recognition for mining truck entry gates.
Uses YOLOv8 (transfer learning from COCO) for detection and ResNet18/34/50 backbone (ImageNet pretrained) + BiLSTM + CTC for recognition.


### Why Transfer Learning (ResNet) instead of training from scratch

- ImageNet-pretrained ResNet already extracts strong visual features (edges, textures, characters).
- Far better generalization on dusty, tilted, low-contrast, and motion-blurred plates.
- Requires significantly less training data and converges much faster.
- Higher final accuracy and robustness compared to a randomly-initialized CRNN.


### Project Structure

iranian_lpr/
├── configs/config.yaml
├── models/
│   ├── detector.py
│   ├── recognizer.py
│   └── pipeline.py
├── utils/
│   ├── image_processing.py
│   ├── plate_utils.py
│   └── database.py
├── inference/
│   └── realtime.py
├── training/
│   ├── train_detector.py
│   └── train_recognizer.py
├── main.py
├── requirements.txt
└── README.md


### 1. Environment Setup

    python -m venv venv
    source venv/bin/activate

    pip install -r requirements.txt

    -Verify CUDA:
    Bashpython -c "import torch; print(torch.cuda.is_available())"


### 2. Prepare Weights

    Bashmkdir -p weights data/ocr_dataset data/plate_dataset
    A. Plate Detector (YOLOv8)

    Collect Iranian plate images (day/night, dusty, tilted, partial occlusion).
    Label class plate.
    Place YOLO-format dataset under data/plate_dataset/ with proper data.yaml.
    Train:

    Bashpython -m training.train_detector
    Best weights are copied automatically to weights/yolov8_plate.pt.
    B. Plate Recognizer (ResNet + CRNN via Transfer Learning)

    Extract plate crops and name files with the exact plate text (e.g. 12ب34567.jpg).
    Place all crops under data/ocr_dataset/.
    Train with ImageNet-pretrained ResNet:

    Bashpython -m training.train_recognizer --config configs/config.yaml
    Weights are saved to weights/resnet_crnn_iran.pt.
    Supported backbones in config: resnet18 (fastest), resnet34, resnet50 (highest accuracy).


### 3. Database

    SQLite is created at data/vehicles.db on first run.
    Demo records are seeded automatically.
    Add real entries:
    Pythonfrom utils.database import VehicleDB
    db = VehicleDB("data/vehicles.db")
    db.upsert("12ب34567", "علی محمدی", "0012345678", "TRK-001", "معدن سنگان", 1)
    4. Configuration
    Edit configs/config.yaml:

    recognizer.backbone: resnet18 / resnet34 / resnet50
    recognizer.pretrained: true (strongly recommended)
    camera.source: 0 or RTSP URL of industrial camera
    thresholds and preprocessing flags


### 5. Run Real-time System

    Bashpython main.py --config configs/config.yaml

    q or Esc to exit.
    Green box = allowed truck.
    Red box = denied / unknown.
    Plate + driver information appear after 3 consecutive stable readings.


### 6. Complete Workflow

    Frame captured from camera.
    YOLOv8 detects plate (robust to tilt and dust).
    Perspective correction + bilateral + CLAHE + sharpen.
    ResNet (ImageNet features) extracts character features.
    BiLSTM + CTC decodes the sequence.
    Iranian plate format validation (XX L YYY ZZ).
    Lookup in vehicle database.
    Overlay on monitor for the gate guard.


### 7. Performance Recommendations

    Start with resnet18 for real-time (>30 FPS on modern GPU).
    Switch to resnet34 or resnet50 if maximum accuracy on heavily degraded plates is required.
    Always keep pretrained: true.
    Collect at least 5k–10k diverse plate crops for the recognizer.
    Use strong geometric and photometric augmentations during detector training.


### 8. Troubleshooting

    Problem,Solution
    Low recognition accuracy,"Train longer, use resnet34/50, add more dusty/tilted crops"
    Slow inference,Use resnet18 + smaller detector (yolov8n)
    Camera not opening,Check source / RTSP / permissions
    Invalid plate format,Verify charset and plate_utils.py regex


### 9. Production Notes

    Run under systemd on the gate computer.
    Point camera.source to fixed industrial camera.
    Keep data/vehicles.db synchronized with central system.
    The entire pipeline is modular – replace only the recognizer weights when you improve the model.

    This version uses full transfer learning on both detection and recognition stages for maximum accuracy and robustness on real-world Iranian mining plates.