🍎 AI Fruit Ripeness Detection & Sorting System
An end-to-end computer vision and robotics system that detects fruit ripeness in real time and sorts them automatically using a SCARA robot arm.

Overview
This project combines deep learning object detection with robotic automation to build a fully autonomous fruit sorting system. A USB camera captures live video of fruits placed on a surface. A custom-trained YOLOv8 model running on a laptop classifies each fruit in real time as ripe, unripe, or defective. Based on the classification, a SCARA robot arm controlled via an Arduino CNC Shield picks the fruit and places it into the correct sorting zone.
Key Highlights

Real-time fruit detection at 5–8 FPS on CPU with 83–98% confidence
Custom YOLOv8n model trained on 706 augmented images across 3 classes
Full pick-and-place automation using 4 NEMA stepper motors via A4988 drivers
Live dashboard showing confidence bars, detection log, FPS counter and sort counts
Automatic CSV logging of every detection with timestamp and centroid coordinates
Complete zone mapping logic — ripe fruits to Zone A, unripe/defective to Zone B


Demo
Live Detection Window
┌─────────────────────────────────────────────┬──────────────────┐
│  ZONE A — Ripe        │  ZONE B — Unripe    │  CONFIDENCE      │
│                       │  /Defective          │  ripe:      91% ██│
│   ┌───────────┐       │                      │  unripe:    12% █ │
│   │ ripe  91% │──────►│                      │  defective:  5%   │
│   │  ●        │       │                      │                   │
│   └───────────┘       │                      │  FPS: 6.2        │
│                       │                      │                   │
│                       │                      │  DETECTION LOG   │
│                       │                      │  11:52:30 RIP ZA │
│                       │                      │  11:52:28 UNR ZB │
├───────────────────────┴──────────────────────│                   │
│  Zone A (Ripe): 4          Zone B: 2         │  Press Q to quit │
└──────────────────────────────────────────────┴──────────────────┘
Terminal Output Sample
[11:52:30] RIPE        | 91% | Centroid (320, 241) | Zone A
[11:52:33] UNRIPE      | 87% | Centroid (298, 255) | Zone B
[11:52:37] DEFECTIVE   | 83% | Centroid (311, 248) | Zone B

Session summary:
  Zone A (Ripe):             4
  Zone B (Unripe/Defective): 2
  Total sorted:              6

Hardware
Components
ComponentSpecificationPurposeLaptopWindows 10/11, any CPURuns Python + YOLOv8USB CameraAny webcam / USB cameraCaptures live fruit videoCamera StandAdjustable mountPositions camera above fruitsSCARA Robot Arm3D printed, 4-axisPick and place sortingArduino UnoATmega328PMotor controllerCNC Shield v3Stacked on ArduinoStepper driver interfaceA4988 Driversx4, with heatsinksDrives each stepper motorNEMA 17 Steppersx4, 1.8° step angleJoint movementPower Supply12V DC adapterPowers stepper motorsUSB-B CableStandard printer cableArduino to laptop
Wiring — CNC Shield Motor Connections
CNC Shield v3 Pin Mapping:
┌─────────────┬──────────┬─────────┬──────────────────┐
│ Motor       │ STEP Pin │ DIR Pin │ Connected To     │
├─────────────┼──────────┼─────────┼──────────────────┤
│ Motor 1 (X) │ Pin 2    │ Pin 5   │ Base rotation    │
│ Motor 2 (Y) │ Pin 3    │ Pin 6   │ Lower arm joint  │
│ Motor 3 (Z) │ Pin 4    │ Pin 7   │ Upper arm joint  │
│ Motor 4 (A) │ Pin 12   │ Pin 13  │ End effector     │
│ Enable (ALL)│ Pin 8    │ —       │ LOW = enabled    │
└─────────────┴──────────┴─────────┴──────────────────┘
Power Requirements

Arduino + CNC Shield: powered via USB from laptop
Stepper motors: require separate 12V DC power supply connected to green screw terminal on CNC Shield
Do NOT run motors on USB power alone — they will not move or will stall


Software
Technology Stack
ToolVersionPurposePython3.10.xCore programming languageUltralytics YOLOv8LatestObject detection modelOpenCV4.8+Camera capture and image processingRoboflowWeb platformDataset annotation and augmentationpyserial3.5+Arduino serial communicationAccelStepperArduino librarySmooth stepper motor controlGit + GitHub—Version control and collaboration
Repository Structure
ai-fruit-ripeness-sorting/
│
├── verify_dataset.py       # Visualize dataset with bounding boxes
├── train.py                # Train YOLOv8 model on custom dataset
├── evaluate.py             # Evaluate model accuracy per class
├── test_images.py          # Run model on test images, save results
├── detect_live.py          # Basic live camera detection
├── zone_mapper.py          # Enhanced live detection with zone logic
├── zone_mapper_v2.py       # Professional version with dashboard UI
├── arm_control.py          # SCARA arm serial control (mock + real)
├── find_port.py            # Auto-detect Arduino COM port
├── test_arm.py             # Test individual arm movements
├── main.py                 # Full integrated pipeline (Day 8)
│
├── data.yaml               # Dataset configuration for YOLOv8
├── requirements.txt        # Python dependencies
├── .gitignore              # Excludes large files from GitHub
└── README.md               # This file

Setup
Prerequisites

Windows 10 or 11 laptop
Python 3.10 installed with "Add to PATH" ticked
Git installed
Arduino IDE 2.x installed

Step 1 — Clone the Repository
bashgit clone https://github.com/PukyBots/ai-fruit-ripeness-sorting.git
cd ai-fruit-ripeness-sorting
Step 2 — Install Python Dependencies
bashpip install ultralytics opencv-python matplotlib pyserial
Step 3 — Download the Dataset
The dataset images are stored on Roboflow (too large for GitHub).

Go to the Roboflow project: Fruit-Ripness on Roboflow
Click Versions → v1 → Export Dataset → YOLOv8 format
Download zip and extract into the project folder
You should see train/, valid/, test/ folders appear

Step 4 — Update data.yaml Paths
Open data.yaml and update the paths to match your laptop:
yamltrain: C:/Users/YOUR_USERNAME/Desktop/ai-fruit-ripeness-sorting/train/images
val:   C:/Users/YOUR_USERNAME/Desktop/ai-fruit-ripeness-sorting/valid/images
test:  C:/Users/YOUR_USERNAME/Desktop/ai-fruit-ripeness-sorting/test/images

nc: 3
names: ['defective', 'ripe', 'unripe']
Find your username by running echo %USERNAME% in Command Prompt.
Step 5 — Get the Trained Model
Option A — Copy from original training laptop:
Copy runs/detect/fruit_ripeness3/weights/best.pt via USB drive to the same path on your laptop.
Option B — Retrain the model:
bashpython train.py
Training takes approximately 1–2 hours on CPU or 20–30 minutes on Google Colab free GPU.
Step 6 — Upload Arduino Firmware

Open Arduino IDE
Go to Sketch → Include Library → Manage Libraries
Search AccelStepper → Install
Open arduino/stepper_test.ino from this repository
Connect Arduino via USB-B cable
Select Tools → Board → Arduino Uno
Select Tools → Port → COM3 (or your detected port)
Click Upload


Usage
Verify Dataset Loads Correctly
bashpython verify_dataset.py
Displays 4 random training images with bounding boxes drawn. Confirms dataset paths are correct.
Train the Model
bashpython train.py
Trains YOLOv8n for 100 epochs on your dataset. Saves best weights to runs/detect/fruit_ripeness3/weights/best.pt.
Evaluate Model Accuracy
bashpython evaluate.py
Prints per-class precision, recall and mAP50 scores on the test set.
Test on Saved Images
bashpython test_images.py
Runs detection on 10 random test images. Saves annotated results to test_results/ folder. Old results are cleared automatically each run.
Live Camera Detection
bashpython detect_live.py
Opens camera and runs real-time detection. Press Q to quit. Press S to save a screenshot.
Zone Mapper — Full Detection Dashboard
bashpython zone_mapper_v2.py
The main detection interface showing:

Live camera feed with bounding boxes and zone arrows
Real-time confidence bars for all 3 classes
Timestamped detection log (last 8 detections)
FPS counter
Running Zone A and Zone B sort counts
Automatic CSV logging of all detections

Press Q to quit. Session summary prints in terminal.
Find Arduino Port
bashpython find_port.py
Plug Arduino in first, then run this. Prints the exact COM port to use in arm_control.py.
Test Robot Arm (Mock Mode)
bashpython arm_control.py
With MOCK_MODE = True (default), simulates all arm movements and prints commands to terminal. Set MOCK_MODE = False and update SERIAL_PORT when real arm is connected.
Full Integrated System
bashpython main.py
Runs the complete pipeline:
Camera captures frame → YOLOv8 detects fruit → Zone decided → Arm picks → Arm places → Repeat

Results
Dataset Statistics
MetricValueSource images photographed294After augmentation (3x)706Training set494 images (70%)Validation set141 images (20%)Test set71 images (10%)Image size512 × 512 pxClasses3 (ripe, unripe, defective)
Augmentation Techniques Applied
TechniqueSettingHorizontal flipEnabledRotation±15°Brightness±20%Gaussian blurUp to 1.5px
Model Performance
ClassPrecisionRecallmAP50Ripe~0.85~0.82~0.83Unripe~0.78~0.74~0.76Defective~0.71~0.68~0.69Overall~0.76
Detection Performance
MetricValueConfidence range on test images83% – 98%Detection speed (CPU)5–8 FPSConfidence threshold used0.50IoU threshold (NMS)0.50

Architecture
System Pipeline
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  USB Camera  │────►│  YOLOv8n     │────►│  Zone Logic  │────►│  SCARA Arm   │
│              │     │  Detection   │     │              │     │              │
│ Captures     │     │ Classifies   │     │ Ripe → A     │     │ Picks fruit  │
│ live frames  │     │ each fruit   │     │ Others → B   │     │ Places in    │
│ 640×480px    │     │ 83-98% conf  │     │ Extracts     │     │ correct zone │
│              │     │              │     │ centroid     │     │              │
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
Zone Decision Logic
pythonif detected_class == 'ripe':
    zone = 'A'          # Left tray — ready for market
else:
    zone = 'B'          # Right tray — unripe or defective

centroid = ((x1+x2)//2, (y1+y2)//2)   # Sent to robot arm
YOLOv8 Model Architecture
The model uses YOLOv8n (nano) with a CSPDarknet CNN backbone for feature extraction. We applied transfer learning — starting from ImageNet pretrained weights and fine-tuning on our custom fruit dataset. This approach requires significantly less data and training time compared to training from scratch while achieving strong accuracy.

Project Timeline
WeekPhaseKey DeliverablesWeek 1Planning & SetupProject scope, hardware list, environment setupWeek 2Data Collection294 fruit photos, Roboflow annotationWeek 3Dataset PreparationAugmentation to 706 images, YOLOv8 exportWeek 4Model TrainingYOLOv8n trained, best.pt saved, evaluationWeek 5Live DetectionCamera pipeline, zone mapping, dashboard UIWeek 6DocumentationGitHub setup, CSV logging, screenshot captureWeek 7Hardware IntegrationArduino firmware, stepper control, full pipeline

Troubleshooting
Camera not opening
Error: Camera not found! Try VideoCapture(1)
Change VideoCapture(0) to VideoCapture(1) in the script. Your USB camera may be index 1 if you also have a built-in webcam.
No bounding boxes appearing
Lower the confidence threshold in the script:
pythonresults = model(frame, conf=0.35, iou=0.5, verbose=False)
Module not found error
bashpip install ultralytics opencv-python matplotlib pyserial
Arduino not detected

Check Device Manager → Ports (COM & LPT)
Install CH340 driver if Arduino is not recognized: CH340 Driver
Run python find_port.py after installing driver

Stepper motors not moving

Confirm 12V power adapter is connected to green screw terminal on CNC Shield
Press A4988 driver boards firmly into shield — loose connection is most common cause
Check EN_PIN is set LOW in Arduino code to enable all drivers

data.yaml path error
Update all 3 paths in data.yaml with your actual Windows username. Find it with:
bashecho %USERNAME%

Contributing
This is an academic project. If you would like to build on this work:

Fork the repository
Create a feature branch: git checkout -b feature/improvement-name
Commit your changes: git commit -m "Add improvement"
Push to the branch: git push origin feature/improvement-name
Open a Pull Request


Team
Fatima Al Ruhala — Computer Vision, Model Training, Zone Mapping Logic, Documentation
Institution: [Your College Name]
Department: [Your Department]
Academic Year: 2025–2026

Acknowledgements

Ultralytics for the YOLOv8 framework
Roboflow for dataset annotation and augmentation tools
AccelStepper Library for smooth stepper motor control
OpenCV for computer vision utilities


License
This project is developed for academic and educational purposes
