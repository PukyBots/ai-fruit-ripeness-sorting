# 🍎 AI-Based Fruit Ripeness Detection and SCARA Robot Pick-and-Place

An AI-powered fruit sorting system that classifies fruits into **Raw**, **Ripe**, and **Defective** categories using computer vision and deep learning. Based on the detected fruit class, a **SCARA robot** performs automated pick-and-place operations for sorting.

The system combines **OpenCV** for real-time image acquisition and preprocessing with a trained AI model for fruit classification, enabling intelligent automation in agricultural and food processing applications.

---

## Features

- Real-time fruit detection using a camera
- AI-based fruit classification
- Classification into:
  - 🟢 Raw
  - 🟡 Ripe
  - 🔴 Defective
- OpenCV-based image processing
- Automated pick-and-place using a SCARA robot
- Real-time decision making
- Modular and scalable architecture

---

## Technologies Used

- Python
- OpenCV
- TensorFlow / Keras (AI Model)
- NumPy
- SCARA Robot
- USB Camera

---

## System Workflow

```text
Camera
   │
   ▼
Image Capture
   │
   ▼
OpenCV Processing
   │
   ▼
AI Classification Model
   │
   ▼
Raw / Ripe / Defective
   │
   ▼
SCARA Robot
   │
   ▼
Pick and Place
```

---

## Dataset Classes

The AI model was trained to classify fruits into three categories:

| Class | Description |
|--------|-------------|
| 🟢 Raw | Unripe fruits |
| 🟡 Ripe | Ready-to-harvest fruits |
| 🔴 Defective | Damaged or spoiled fruits |

---

## Project Structure

```text
.
├── dataset/
│   ├── raw/
│   ├── ripe/
│   └── defective/
│
├── training/
│   ├── train_model.py
│   └── model/
│
├── detection/
│   ├── camera_detection.py
│   └── opencv_processing.py
│
├── scara_robot/
│   └── robot_control.py
│
├── images/
│
├── README.md
└── requirements.txt
```

---

## How It Works

1. Capture fruit images using a camera.
2. Process the images using OpenCV.
3. Pass the processed image to the trained AI model.
4. Classify the fruit as **Raw**, **Ripe**, or **Defective**.
5. Send the classification result to the SCARA robot.
6. The SCARA robot performs the corresponding pick-and-place operation.

---

## Applications

- Smart Agriculture
- Fruit Grading
- Food Processing Industries
- Packaging Automation
- Quality Inspection
- Industrial Pick-and-Place Systems

---

## Future Improvements

- Multi-fruit detection in a single frame
- Fruit size estimation
- Ripeness percentage prediction
- Conveyor belt integration
- Real-time production statistics dashboard
- Support for additional fruit varieties
- Edge AI deployment on embedded devices

---

## Contributions

This project was developed by **Pulkit Garg**.

Additional contributions were made by **Yenepoya University students** as part of their training:

- **Fathimathul Risa T.P.** – Contributed to dataset preparation, image annotation, and AI model training.
- **Fatima Al Ruhala** – Contributed to dataset preparation, image annotation, AI model training, and development of the SCARA robot stepper motor control code.

---

## License

This project is intended for educational, research, and automation purposes.
