from ultralytics import YOLO
import cv2, os, random, shutil, subprocess

model = YOLO(r'C:\Users\admin\Desktop\fruit_project\runs\detect\train\weights\best.pt')

test_folder   = r'C:\Users\admin\Desktop\fruit_project\test\images'
output_folder = r'C:\Users\admin\Desktop\fruit_project\test_results'

# Force delete and recreate
if os.path.exists(output_folder):
    shutil.rmtree(output_folder, ignore_errors=True)
os.makedirs(output_folder, exist_ok=True)
print("✓ Cleared old results")

CLASS_NAMES = ['defective', 'ripe', 'unripe']
COLORS = {
    'defective': (0, 0, 255),
    'ripe':      (0, 255, 0),
    'unripe':    (0, 165, 255),
}

images = [f for f in os.listdir(test_folder) if f.endswith(('.jpg','.png'))]
sample = random.sample(images, min(10, len(images)))
print(f"✓ Testing {len(sample)} fresh images\n")

for img_file in sample:
    img_path = os.path.join(test_folder, img_file)
    img = cv2.imread(img_path)
    results = model(img_path, conf=0.6, iou=0.5)

    count = 0
    for box in results[0].boxes:
        cls  = CLASS_NAMES[int(box.cls[0])]
        conf = float(box.conf[0])
        x1,y1,x2,y2 = map(int, box.xyxy[0])
        color = COLORS[cls]
        cv2.rectangle(img, (x1,y1), (x2,y2), color, 2)
        label = f"{cls} {conf:.0%}"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (x1, y1-25), (x1+w, y1), color, -1)
        cv2.putText(img, label, (x1, y1-7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        print(f"  {cls} {conf:.0%}")
        count += 1

    if count == 0:
        print(f"  {img_file[:20]} → no detection")

    cv2.imwrite(os.path.join(output_folder, img_file), img)
    print(f"  Saved: {img_file}\n")

print("=" * 40)
print(f"Done! Opening results folder...")