import cv2
import time
from ultralytics import YOLO

model = YOLO(r'C:\Users\admin\Desktop\fruit_project\best.pt')

CLASS_NAMES = ['defective', 'ripe', 'unripe']
COLORS = {
    'defective': (0, 0, 255),
    'ripe':      (0, 255, 0),
    'unripe':    (0, 165, 255),
}

cap = cv2.VideoCapture(2)  # 1 = external webcam, use 0 for laptop camera
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("Camera not found! Try changing VideoCapture index (0, 1, 2...)")
    exit()

print("Running — press Q to quit")

prev_time = time.time()
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.5, iou=0.5, verbose=False)

    detected = []
    for box in results[0].boxes:
        cls  = CLASS_NAMES[int(box.cls[0])]
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        color = COLORS[cls]

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label = f"{cls} {conf:.0%}"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(frame, (x1, y1 - 28), (x1 + w + 10, y1), color, -1)
        cv2.putText(frame, label, (x1 + 5, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        detected.append(f"{cls}({conf:.0%})")

    # FPS calculation
    frame_count += 1
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    # Overlay info
    cv2.putText(frame, f"FPS: {fps:.1f}",            (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, f"Detected: {len(detected)}",  (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, "Press Q to quit",             (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # Log to terminal every 30 frames
    if frame_count % 30 == 0 and detected:
        print(f"Frame {frame_count}: {', '.join(detected)}")

    cv2.imshow("Fruit Ripeness Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Done!")