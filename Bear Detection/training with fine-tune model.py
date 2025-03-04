from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2
import numpy as np
import time
import pandas as pd
from collections import defaultdict, deque

# Load YOLO Model
model = YOLO("/home/realtimeidns/best2.pt")

# Initialize DeepSORT Tracker with tuned parameters
tracker = DeepSort(max_age=100, nn_budget=70, max_iou_distance=0.5)

# Constants
FRAME_SKIP = 2  
OUTPUT_VIDEO_PATH = "op_w_FT.mp4"
CONFIDENCE_THRESHOLD = 0.7  # Lowered for better sensitivity
STABILITY_THRESHOLD = 0.2  # If stability score > 0.2, trigger alert
MIN_FRAME_PERSISTENCE = 3  # Reduced from 5 to 3

# List of Wild Animal Class IDs
WILD_ANIMALS = {0: "bear", 8: "wolf"}  

# Open Video
video_path = "/home/realtimeidns/Downloads/Bear catches camera.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(" ERROR: Unable to open video file.")
    exit()

frame_id = 0
total_processing_time = 0

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

if fps == 0 or frame_width == 0 or frame_height == 0:
    print(" ERROR: Video properties are invalid. Check your input file.")
    exit()

# Ensure FPS is valid
output_fps = max(1, fps // FRAME_SKIP)

# Define video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, output_fps, (frame_width, frame_height))

print(f"🎥 Processing video: {video_path}")
print(f"Video FPS: {fps}, Frame Size: {frame_width}x{frame_height}, Output FPS: {output_fps}")

# Tracking history & object persistence
object_confidences = defaultdict(lambda: deque(maxlen=10))  # Store last 10 confidence scores
object_persistence = defaultdict(int)
bear_alert_triggered = False

# OpenCV Optimization
cv2.setUseOptimized(True)

# Process Video Frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print(" Video processing completed.")
        break  # Stop if no more frames

    if frame_id % FRAME_SKIP != 0:
        frame_id += 1
        continue

    start_time = time.time()

    # Run YOLO Detection
    results = model(frame)
    detections = results[0].boxes.xyxy.cpu().numpy()
    scores = results[0].boxes.conf.cpu().numpy()
    classes = results[0].boxes.cls.cpu().numpy()

    for bbox, score, cls_id in zip(detections, scores, classes):
        if score >= CONFIDENCE_THRESHOLD:
            class_id = int(cls_id)

            # Only track wild animals
            if class_id in WILD_ANIMALS:
                object_confidences[class_id].append(score)
                object_persistence[class_id] += 1

                # Draw Bounding Box
                x1, y1, x2, y2 = map(int, bbox)
                label = f"{WILD_ANIMALS[class_id]} ({score:.2f})"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Write processed frame to output video
    out.write(frame)

    frame_id += 1
    total_processing_time += time.time() - start_time

# Release resources
cap.release()
out.release()

# Compute Stability Score
total_tracked_frames = min(100, frame_id)  # Rolling window of last 100 frames
object_stability_scores = {
    cls_id: (np.mean(confidences) * object_persistence[cls_id]) / total_tracked_frames
    for cls_id, confidences in object_confidences.items()
    if object_persistence[cls_id] > 0
}

# Save stability scores to CSV
df_scores = pd.DataFrame(list(object_stability_scores.items()), columns=["Class_ID", "Stability_Score"])
df_scores.to_csv("object_stability_scores_refined.csv", index=False)

print(" Object stability scores computed and saved!")

# Check if we need to raise an alert
alert_triggered = False
for cls_id, stability_score in object_stability_scores.items():
    if (stability_score > STABILITY_THRESHOLD and cls_id in WILD_ANIMALS) or \
       (cls_id == 0 and object_persistence[cls_id] >= MIN_FRAME_PERSISTENCE and np.mean(object_confidences[cls_id]) > 0.5):
        print(f" ALERT: {WILD_ANIMALS[cls_id].upper()} detected with stability score {stability_score:.3f}! ")
        alert_triggered = True
        bear_alert_triggered = True

if not alert_triggered:
    print(" No wild animal detected above the threshold. No alert raised.")
