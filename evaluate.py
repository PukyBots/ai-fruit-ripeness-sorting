from ultralytics import YOLO

# Full absolute path to the model
model = YOLO(r'C:\Users\admin\Desktop\fruit_project\runs\detect\train\weights\best.pt')

# Run validation on test split
metrics = model.val(
    data=r'C:\Users\admin\Desktop\fruit_project\data.yaml',
    split='test'
)

# Define class names
class_names = ['defective', 'ripe', 'unripe']

print("\n===== RESULTS PER CLASS =====")
for i, name in enumerate(class_names):
    p = metrics.box.p[i]
    r = metrics.box.r[i]
    m = metrics.box.ap50[i]
    print(f"{name:12s} | Precision: {p:.2f} | Recall: {r:.2f} | mAP50: {m:.2f}")

print(f"\nOverall mAP50: {metrics.box.map50:.4f}")
print("=============================")
