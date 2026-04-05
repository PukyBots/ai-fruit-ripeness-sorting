import cv2, time
from ultralytics import YOLO
from collections import deque
import datetime
import csv
import os

model = YOLO(r'C:\Users\admin\Desktop\fruit_project\best.pt')
CLASS_NAMES = ['defective', 'ripe', 'unripe']
COLORS = {
    'defective': (0,   0,   255),
    'ripe':      (0,   255, 0),
    'unripe':    (0,   165, 255),
}
BAR_COLORS = {
    'defective': (60,  60,  220),
    'ripe':      (40,  180, 40),
    'unripe':    (30,  140, 220),
}

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("Camera not found! Try VideoCapture(1)")
    exit()

zone_a_count  = 0
zone_b_count  = 0
detection_log = deque(maxlen=8)
conf_bars     = {'defective': 0.0, 'ripe': 0.0, 'unripe': 0.0}
last_cls      = None
last_det_time = 0
prev_time     = time.time()

print("Enhanced Zone Mapper running — press Q to quit")

csv_filename = f"detection_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), csv_filename)
csv_file = open(csv_path, 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['Timestamp', 'Class', 'Confidence', 'Centroid_X', 'Centroid_Y', 'Zone', 'Session_ZoneA', 'Session_ZoneB'])
print(f"Logging detections to: {csv_filename}")

screenshots_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'screenshots')
os.makedirs(screenshots_folder, exist_ok=True)
screenshot_count = 0
print(f"Press S to save screenshot — saved to: screenshots/")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time + 1e-6)
    prev_time = curr_time

    panel_x = w - 240
    cv2.rectangle(frame, (panel_x, 0), (w, h), (30, 30, 30), -1)
    cv2.line(frame, (panel_x, 0), (panel_x, h), (80, 80, 80), 1)

    mid_x = panel_x // 2
    cv2.line(frame, (mid_x, 40), (mid_x, h - 50), (200, 200, 200), 2)

    cv2.rectangle(frame, (0, 0), (mid_x, 38), (0, 140, 0), -1)
    cv2.putText(frame, "ZONE A — Ripe",
                (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.rectangle(frame, (mid_x, 0), (panel_x, 38), (180, 90, 0), -1)
    cv2.putText(frame, "ZONE B — Unripe/Defective",
                (mid_x + 6, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    results = model(frame[:, :panel_x], conf=0.5, iou=0.5, verbose=False)

    for k in conf_bars:
        conf_bars[k] = 0.0

    for box in results[0].boxes:
        cls  = CLASS_NAMES[int(box.cls[0])]
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        color = COLORS[cls]
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        zone = 'A' if cls == 'ripe' else 'B'

        conf_bars[cls] = max(conf_bars[cls], conf)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.circle(frame, (cx, cy), 6, (255, 255, 255), -1)
        cv2.circle(frame, (cx, cy), 4, color, -1)

        label = f"{cls} {conf:.0%} -> Zone {zone}"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(frame, (x1, y1 - 26), (x1 + lw + 8, y1), color, -1)
        cv2.putText(frame, label, (x1 + 4, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        ax = 80 if zone == 'A' else panel_x - 80
        cv2.arrowedLine(frame, (cx, cy), (ax, cy),
                        (0, 200, 0) if zone == 'A' else (0, 130, 220),
                        2, tipLength=0.3)

        if cls != last_cls or curr_time - last_det_time > 2.0:
            ts = datetime.datetime.now().strftime("%H:%M:%S")
            detection_log.appendleft((ts, cls, conf, zone))
            if zone == 'A':
                zone_a_count += 1
            else:
                zone_b_count += 1
            last_cls      = cls
            last_det_time = curr_time
            print(f"[{ts}] {cls.upper():10s} | {conf:.0%} | Centroid ({cx},{cy}) | Zone {zone}")
            csv_writer.writerow([ts, cls, f"{conf:.2f}", cx, cy, zone, zone_a_count, zone_b_count])
            csv_file.flush()

    cv2.putText(frame, "CONFIDENCE",
                (panel_x + 10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    bar_y = 50
    for cls_name, conf_val in conf_bars.items():
        bcolor = BAR_COLORS[cls_name]
        bar_w  = int(conf_val * 200)
        cv2.rectangle(frame,
                      (panel_x + 10, bar_y),
                      (panel_x + 230, bar_y + 22),
                      (60, 60, 60), -1)
        if bar_w > 0:
            cv2.rectangle(frame,
                          (panel_x + 10, bar_y),
                          (panel_x + 10 + bar_w, bar_y + 22),
                          bcolor, -1)
        cv2.putText(frame, f"{cls_name}: {conf_val:.0%}",
                    (panel_x + 14, bar_y + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)
        bar_y += 30

    cv2.putText(frame, f"FPS: {fps:.1f}",
                (panel_x + 10, bar_y + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    log_y = bar_y + 40
    cv2.putText(frame, "DETECTION LOG",
                (panel_x + 10, log_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    log_y += 18
    cv2.line(frame, (panel_x + 10, log_y), (w - 10, log_y), (80, 80, 80), 1)
    log_y += 8

    for ts, det_cls, det_conf, det_zone in detection_log:
        zclr = (40, 200, 40) if det_zone == 'A' else (30, 140, 220)
        cv2.putText(frame, f"{ts} {det_cls[:3].upper()} Z{det_zone}",
                    (panel_x + 10, log_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, zclr, 1)
        log_y += 18
        if log_y > h - 60:
            break

    cv2.rectangle(frame, (0, h - 44), (panel_x, h), (40, 40, 40), -1)
    cv2.putText(frame, f"Zone A (Ripe): {zone_a_count}",
                (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 220, 40), 2)
    cv2.putText(frame, f"Zone B (Unripe/Def): {zone_b_count}",
                (mid_x + 10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 150, 220), 2)
    cv2.putText(frame, "Press Q to quit",
                (10, h - 52), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

    cv2.imshow("Fruit Ripeness Sorting System", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        screenshot_count += 1
        ts_file  = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"screenshot_{ts_file}_{screenshot_count}.jpg"
        filepath = os.path.join(screenshots_folder, filename)
        cv2.imwrite(filepath, frame)
        print(f"Screenshot saved: {filename}")
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
csv_file.close()
print(f"\nDetection log saved to: {csv_path}")
print(f"\nSession summary:")
print(f"  Zone A (Ripe):             {zone_a_count}")
print(f"  Zone B (Unripe/Defective): {zone_b_count}")
print(f"  Total sorted:              {zone_a_count + zone_b_count}")