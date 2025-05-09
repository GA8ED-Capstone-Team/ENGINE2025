import torch
import numpy as np
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.current_device())
import cv2 
import torch
from torch.serialization import add_safe_globals
from ultralytics.nn.tasks import DetectionModel
add_safe_globals([DetectionModel]) # Add the class to the trusted globals
from ultralytics import YOLO
from ultralytics import SAM

print(cv2.__version__)
model = YOLO("yolo11x.pt")
sam_model = SAM("sam2.1_b.pt")

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use 0 for the default camera, or provide a video file path
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame.")
        break
    results = model.predict(frame, conf=0.5)
    result = results[0]
    boxes = result.boxes
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        w, h = int(x2)-int(x1), int(y2)-int(y1)
        class_number = int(box.cls[0].cpu().numpy())
        conf = float(box.conf[0].cpu().numpy())

        #help(sam_model.predict)

        
        sam_result = sam_model.predict(frame, bboxes=[x1, y1, x2, y2])
        sam_mask = sam_result[0].masks.data[0].cpu().numpy().astype(bool)
        colored_mask = np.zeros_like(frame, dtype=np.uint8)
        colored_mask[sam_mask] = [0, 0, 255]  # Red color for the mask
        
        frame = cv2.addWeighted(frame, 0.5, colored_mask, 0.5, 0)

        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, f"Class: {class_number}, Conf: {conf:.2f}", (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    cv2.imshow("YOLO Detection + SAM", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
