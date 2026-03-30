from ultralytics import YOLO

model = YOLO('yolov8n.pt')
results = model.train(
    data=r'C:\Users\admin\Desktop\fruit_project\data.yaml',  # ← exact path to data.yaml
    epochs=100,
    imgsz=512,
    batch=8,
    name='fruit_ripeness',
    patience=20,
    save=True,
    plots=True
)

print("Training complete!")
print("Model saved at: runs/detect/fruit_ripeness/weights/best.pt")