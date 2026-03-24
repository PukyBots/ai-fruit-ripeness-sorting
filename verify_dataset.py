import cv2
import os
import random
import matplotlib.pyplot as plt

train_img = r"paste the path here"
train_lbl = r"paste the train data path here"

images = os.listdir(train_img)
labels = os.listdir(train_lbl)
print(f"Total train images: {len(images)}")
print(f"Total train labels: {len(labels)}")

class_names = ['defective', 'ripe', 'unripe']
colors = [(0,0,255), (0,255,0), (255,165,0)]

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
sample = random.sample(images, 4)

for i, img_file in enumerate(sample):
    img_path = os.path.join(train_img, img_file)
    lbl_path = os.path.join(train_lbl, img_file.replace('.jpg', '.txt'))
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    if os.path.exists(lbl_path):
        with open(lbl_path) as f:
            for line in f:
                cls, cx, cy, bw, bh = map(float, line.split())
                x1 = int((cx-bw/2)*w); y1 = int((cy-bh/2)*h)
                x2 = int((cx+bw/2)*w); y2 = int((cy+bh/2)*h)
                color = colors[int(cls)]
                cv2.rectangle(img,(x1,y1),(x2,y2),color,2)
                cv2.putText(img,class_names[int(cls)],(x1,y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,color,2)
    axes[i].imshow(img)
    axes[i].axis('off')
    axes[i].set_title(img_file[:15])

plt.tight_layout()
plt.savefig("dataset_check.png")
plt.show()
print("Done!")
