import cv2
import numpy as np
import time
from ultralytics import YOLO
from pymycobot.mycobot280 import MyCobot280
from pymycobot import PI_PORT, PI_BAUD

# ------------- USER PARAMETERS -------------
MODEL_PATH = "best.pt"  # Your trained YOLOv8 weights
TABLE_Z = 0.0           # Height of the table surface
PRE_PICK_Z = 150.0      # Hover height before picking
PRE_DROP_Z = 150.0      # Hover height over the box lids
robot_busy = False

# ------------- BOX LID POSITIONS (X, Y, Z) -------------
# You will need to calibrate these! (Instructions below)
BOX_POSITIONS = {
    "ripe": [270.3, -100.7, 56.9],      # Zone A (Left lid)
    "unripe": [279.0, 76.3, 55.8],      # Zone B (Right lid)
    "defective": [279.0, 76.3, 55.8]    # Zone B (Right lid)
}

# ------------- HOMOGRAPHY CALIBRATION -------------
# You MUST calibrate these 4 points for your specific table setup!
pixels = np.array([
    [233, 50],   # Camera pixel Point 1
    [210, 226],  # Camera pixel Point 2
    [122, 44],   # Camera pixel Point 3
    [95, 215],   # Camera pixel Point 4
], dtype=float)

world_coords = np.array([
    [115, -111],    # Robot physical Point 1
    [122.9, 76.9],  # Robot physical Point 2
    [216.4, -120],  # Robot physical Point 3
    [236.0, 80.1],  # Robot physical Point 4
], dtype=float)

H, _ = cv2.findHomography(pixels, world_coords)

# ------------- INIT ROBOT -------------
print("[INFO] Connecting to robot...")
mc = MyCobot280(PI_PORT, PI_BAUD)
mc.power_on()
time.sleep(2)

# ------------- INIT YOLO MODEL -------------
print("[INFO] Loading YOLO model...")
model = YOLO(MODEL_PATH)

# ------------- CAMERA INIT -------------
vs = cv2.VideoCapture(0)
vs.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
vs.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# ------------- HELPER FUNCTIONS -------------
def pixel_to_table(uv, H, table_z=TABLE_Z):
    uv_h = np.array([uv[0], uv[1], 1.0])
    XY_h = H @ uv_h
    XY_h /= XY_h[2]
    X, Y = XY_h[0], XY_h[1]
    return np.array([X, Y, table_z])

def pick_object(target_xyz):
    # 1. Open Gripper
    mc.set_gripper_value(100, 50)
    time.sleep(1)
    # 2. Hover above fruit
    mc.send_coords([target_xyz[0], target_xyz[1], PRE_PICK_Z, -170.0, 0, 0], 50, 0)
    time.sleep(2)
    # 3. Drop down to fruit
    mc.send_coords([target_xyz[0], target_xyz[1], target_xyz[2], -170.0, 0, 0], 30, 0)
    time.sleep(2)
    # 4. Close Gripper
    mc.set_gripper_value(0, 50)
    time.sleep(1)
    # 5. Lift up
    mc.send_coords([target_xyz[0], target_xyz[1], PRE_PICK_Z, -170.0, 0, 0], 50, 0)
    time.sleep(2)

def place_object(box_xyz):
    # 1. Hover over the box lid
    mc.send_coords([box_xyz[0], box_xyz[1], PRE_DROP_Z, -170.0, 0, 0], 50, 0)
    time.sleep(2)
    # 2. Lower into the lid slightly
    mc.send_coords([box_xyz[0], box_xyz[1], box_xyz[2], -170.0, 0, 0], 30, 0)
    time.sleep(2)
    # 3. Open Gripper
    mc.set_gripper_value(100, 50)
    time.sleep(1)
    # 4. Lift up
    mc.send_coords([box_xyz[0], box_xyz[1], PRE_DROP_Z, -170.0, 0, 0], 50, 0)
    time.sleep(2)

def return_home():
    mc.send_angles([0, 0, 0, 0, 0, 0], 50)
    time.sleep(2)

# Move home initially
return_home()

# ------------- MAIN LOOP -------------
while True:
    if not robot_busy:
        ret, frame = vs.read()
        if not ret:
            break

        # Run YOLO inference
        results = model(frame, conf=0.5, iou=0.5, verbose=False)
        
        target_fruit_class = None
        best_cx, best_cy = None, None

        # Process YOLO detections
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                class_name = model.names[cls_id].lower()

                # Calculate center of the bounding box
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                # Just grab the first confident fruit we see
                if best_cx is None:
                    best_cx, best_cy = cx, cy
                    target_fruit_class = class_name
                
                # Draw visual feedback
                color = (0, 255, 0) if class_name == "ripe" else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)
                cv2.putText(frame, f"{class_name.upper()} ({cx}, {cy})", (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # If a fruit was found, trigger the robot
        if target_fruit_class and best_cx is not None:
            robot_busy = True
            
            # 1. Translate pixel to physical millimeters
            target_xyz = pixel_to_table((best_cx, best_cy), H)
            print(f"[ACTION] Found {target_fruit_class} at {target_xyz}. Picking...")
            
            # 2. Pick it up
            pick_object(target_xyz)
            
            # 3. Place it in the correct lid
            box_xyz = BOX_POSITIONS.get(target_fruit_class)
            if box_xyz:
                # Custom Print Logic for the Zones
                if target_fruit_class == "ripe":
                    print("[ACTION] Sorting to Zone A (ripe) tray...")
                elif target_fruit_class in ["unripe", "defective"]:
                    print("[ACTION] Sorting to Zone B (unripe/defective) tray...")
                
                place_object(box_xyz)
                
            # 4. Reset
            return_home()
            robot_busy = False 

    cv2.imshow("Fruit Ripeness Sorting", frame)
    key = cv2.waitKey(1)
    if key == 27:  # ESC key
        mc.release_all_servos()
        break

vs.release()
cv2.destroyAllWindows()