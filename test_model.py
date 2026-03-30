from ultralytics import YOLO
import cv2, os

# ✅ best.pt is directly inside fruit_project folder
model = YOLO(r'C:\Users\admin\Desktop\fruit_project\best.pt')

test_folder = r'C:\Users\admin\Desktop\fruit_project\test\images'

img_file = os.listdir(test_folder)[0]
img_path = os.path.join(test_folder, img_file)

results = model(img_path)
results[0].save(filename='test_result.jpg')
print("Done! Check test_result.jpg in your folder.")
for img_file in os.listdir(test_folder):
    img_path = os.path.join(test_folder, img_file)
    results = model(img_path)
for r in results:
    for box in r.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        names = ['Defective', 'Ripe', 'Unripe']
        print(f"Detected: {names[cls]} — Confidence: {conf:.2f}")

