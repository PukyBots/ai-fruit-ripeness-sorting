import sys
import time
import cv2
from dataclasses import dataclass
from typing import Tuple
from ultralytics import YOLO


CLASS_NAMES = ["defective", "ripe", "unripe"]

CLASS_COLORS: dict[str, Tuple[int, int, int]] = {
    "defective": (0,   0,   255),
    "ripe":      (0,   255, 0  ),
    "unripe":    (0,   165, 255),
}

ZONE_A_COLOR = (0, 200, 0)
ZONE_B_COLOR = (0, 130, 200)

DEBOUNCE_SECONDS = 3
CONFIRM_FRAMES   = 7
MIN_CONFIDENCE   = 0.75


@dataclass
class Config:
    model_path:   str   = r"C:\Users\admin\Desktop\fruit_project\runs\detect\train\weights\best.pt"
    camera_index: int   = 2
    frame_width:  int   = 640
    frame_height: int   = 480
    conf_thresh:  float = 0.75
    iou_thresh:   float = 0.5


def draw_hud(frame, mid_x: int, zone_a_count: int, zone_b_count: int, status_msg: str) -> None:
    h, w = frame.shape[:2]

    cv2.line(frame, (mid_x, 0), (mid_x, h), (200, 200, 200), 2)

    cv2.rectangle(frame, (0, 0), (mid_x, 40), (0, 180, 0), -1)
    cv2.putText(frame, "ZONE A — Ripe",
        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.rectangle(frame, (mid_x, 0), (w, 40), (0, 130, 200), -1)
    cv2.putText(frame, "ZONE B — Unripe / Defective",
        (mid_x + 10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.rectangle(frame, (0, h-80), (w, h), (40, 40, 40), -1)

    cv2.putText(frame, f"Zone A (Ripe): {zone_a_count}", (10, h-55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(frame, f"Zone B (Unripe/Defective): {zone_b_count}", (w//2+10, h-55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

    cv2.putText(frame, status_msg, (10, h-25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 100), 2)

    cv2.putText(frame, "Press Q to quit",
        (10, h - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)


def draw_detection(frame, x1, y1, x2, y2, cx, cy, cls, conf, zone, zone_color, frame_width) -> None:
    color = CLASS_COLORS[cls]

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.circle(frame, (cx, cy), 6, (255, 255, 255), -1)
    cv2.circle(frame, (cx, cy), 4, color, -1)

    label = f"{cls} ({conf:.0%}) -> Zone {zone}"
    (lw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (x1, y1 - 30), (x1 + lw + 10, y1), color, -1)
    cv2.putText(frame, label,
        (x1 + 5, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    arrow_x = 100 if zone == "A" else frame_width - 100
    cv2.arrowedLine(frame, (cx, cy), (arrow_x, cy), zone_color, 2, tipLength=0.3)


def classify_zone(cls: str) -> Tuple[str, Tuple[int, int, int]]:
    if cls == "ripe":
        return "A", ZONE_A_COLOR
    return "B", ZONE_B_COLOR


def process_frame(
    frame, model, cfg: Config,
    zone_a_count: int, zone_b_count: int,
    consecutive_count: int, pending_cls: str, pending_zone: str,
    last_counted_time: float, last_counted_cls: str,
    last_detection: str, last_detection_time: float
):
    h, w = frame.shape[:2]
    mid_x = w // 2
    status_msg = "Waiting for fruit..."

    results = model(frame, conf=cfg.conf_thresh, iou=cfg.iou_thresh, verbose=False)

    detected_this_frame = False

    for box in results[0].boxes:
        cls  = CLASS_NAMES[int(box.cls[0])]
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        zone, zone_color = classify_zone(cls)

        draw_detection(frame, x1, y1, x2, y2, cx, cy, cls, conf, zone, zone_color, w)
        detected_this_frame = True

        # Check if same class as pending
        if cls == pending_cls:
            consecutive_count += 1
        else:
            consecutive_count = 1
            pending_cls  = cls
            pending_zone = zone

        now = time.time()
        time_since_last = now - last_counted_time
        is_same_as_last = (cls == last_counted_cls)

        if consecutive_count >= CONFIRM_FRAMES:
            if not is_same_as_last or time_since_last >= DEBOUNCE_SECONDS:

                # Count and log every new detection
                curr_time = time.time()
                if (cls != last_detection or curr_time - last_detection_time > 2):
                    if zone == 'A':
                        zone_a_count += 1
                    else:
                        zone_b_count += 1

                    last_detection      = cls
                    last_detection_time = curr_time

                    print(
                        f"Detected: {cls.upper():10s} | "
                        f"Centroid: ({cx:3d}, {cy:3d}) | "
                        f"Confidence: {conf:.0%} | "
                        f"-> MOVE TO ZONE {zone} "
                        f"[A:{zone_a_count} B:{zone_b_count}]"
                    )

                last_counted_time = now
                last_counted_cls  = cls
                consecutive_count = 0
                status_msg = f"✔ Counted {cls} → Zone {zone}"

            else:
                remaining  = DEBOUNCE_SECONDS - time_since_last
                status_msg = f"Hold... remove fruit and show next ({remaining:.1f}s)"
        else:
            status_msg = f"Confirming {cls}... ({consecutive_count}/{CONFIRM_FRAMES})"

        break  # Only process the highest-confidence detection per frame

    if not detected_this_frame:
        consecutive_count = 0
        pending_cls  = ""
        status_msg   = "Waiting for fruit..."

    draw_hud(frame, mid_x, zone_a_count, zone_b_count, status_msg)

    return (zone_a_count, zone_b_count,
            consecutive_count, pending_cls, pending_zone,
            last_counted_time, last_counted_cls,
            last_detection, last_detection_time)


def open_camera(cfg: Config) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(cfg.camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  cfg.frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.frame_height)

    if not cap.isOpened():
        print(f"[ERROR] Camera index {cfg.camera_index} not found. Try camera_index=1.")
        sys.exit(1)

    return cap


def main() -> None:
    cfg   = Config()
    model = YOLO(cfg.model_path)
    cap   = open_camera(cfg)

    print("Zone Mapper running — press Q to quit")

    zone_a_count        = 0
    zone_b_count        = 0
    consecutive_count   = 0
    pending_cls         = ""
    pending_zone        = ""
    last_counted_time   = 0.0
    last_counted_cls    = ""
    last_detection      = ""
    last_detection_time = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARNING] Failed to read frame — camera disconnected?")
                break

            (zone_a_count, zone_b_count,
             consecutive_count, pending_cls, pending_zone,
             last_counted_time, last_counted_cls,
             last_detection, last_detection_time) = process_frame(
                frame, model, cfg,
                zone_a_count, zone_b_count,
                consecutive_count, pending_cls, pending_zone,
                last_counted_time, last_counted_cls,
                last_detection, last_detection_time
            )

            cv2.imshow("Zone Mapper — Fruit Sorting", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"\nZone Mapper stopped.")
        print(f"Final Count — Zone A (Ripe): {zone_a_count} | Zone B (Unripe/Defective): {zone_b_count}")


if __name__ == "__main__":
    main()