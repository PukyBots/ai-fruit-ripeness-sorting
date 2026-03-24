import cv2
import os

# 1. Create the 3 categories automatically
for folder in ['ripe', 'unripe', 'defective']:
    os.makedirs(folder, exist_ok=True)

# 2. Try index 1 for USB Webcam (If it shows DroidCam, change to 2)
cap = cv2.VideoCapture(1) 

print("--- SYSTEM ACTIVE ---")
print("1. CLICK the 'Video_Feed' window with your mouse!")
print("2. Press 'r' for Ripe | 'u' for Unripe | 'd' for Defective")
print("3. Press 'q' to Quit")

counts = {'ripe': 0, 'unripe': 0, 'defective': 0}

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Camera not found. Try changing the number in the code.")
        break

    cv2.imshow("Video_Feed", frame)
    key = cv2.waitKey(1) & 0xFF

    # Saving Logic
    if key == ord('r'):
        counts['ripe'] += 1
        cv2.imwrite(f"ripe/img_{counts['ripe']}.jpg", frame)
        print(f"Captured Ripe: {counts['ripe']}/100")

    elif key == ord('u'):
        counts['unripe'] += 1
        cv2.imwrite(f"unripe/img_{counts['unripe']}.jpg", frame)
        print(f"Captured Unripe: {counts['unripe']}/100")

    elif key == ord('d'):
        counts['defective'] += 1
        cv2.imwrite(f"defective/img_{counts['defective']}.jpg", frame)
        print(f"Captured Defective: {counts['defective']}/100")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Collection Complete.")
