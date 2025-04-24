#  Cell 1: Import libraries
import os
import cv2
import base64
import uuid
import google.generativeai as genai
from ultralytics import YOLO

#  Cell 2: Configuration
VIDEO_PATH = "/home/realtimeidns/Downloads/llmvand.mp4"
MODEL_PATH = "/home/realtimeidns/Vandalism/yolo11x.pt"
FRAME_OUTPUT_DIR = "frames_for_genai"
CONFIDENCE_THRESHOLD = 0.25  # More sensitive detection
FRAME_SKIP = 2
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or "AIzaSyBTIbICpwJKEcyqSc0fydk8hl1BR6qzkRI"

#  Cell 3: Setup environment
os.makedirs(FRAME_OUTPUT_DIR, exist_ok=True)
genai.configure(api_key=GEMINI_API_KEY)
model = YOLO(MODEL_PATH)
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

#  Cell 4: Main loop
cap = cv2.VideoCapture(VIDEO_PATH)
frame_id = 0
fps = int(cap.get(cv2.CAP_PROP_FPS))
print(f" Scanning video: {VIDEO_PATH}")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_id % FRAME_SKIP != 0:
        frame_id += 1
        continue

    results = model.predict(source=frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
    if not results:
        frame_id += 1
        continue

    detections = results[0].boxes.xyxy.cpu().numpy()
    scores = results[0].boxes.conf.cpu().numpy()
    classes = results[0].boxes.cls.cpu().numpy()
    class_names = [model.names[int(cls)] for cls in classes]

    if "car" in class_names:
        filename = f"frame_{uuid.uuid4().hex[:8]}.jpg"
        frame_path = os.path.join(FRAME_OUTPUT_DIR, filename)
        cv2.imwrite(frame_path, frame)
        print(f" Saved frame: {frame_path}")

        with open(frame_path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode("utf-8")

        prompt_parts = [
            {
                "inline_data": {
                    "mime_type": "image/jpeg",
                    "data": image_b64
                }
            },
            "You are a neighborhood safety assistant. Does this image show someone damaging a car—scratching, kicking, or breaking it? If yes, respond with: 'Yes: [action]'. If not, say 'No: [reason]'."
        ]

        try:
            response = gemini_model.generate_content(prompt_parts)
            print(f" Gemini Response:\n{response.text}\n")
        except Exception as e:
            print(f" Gemini API error: {e}\n")

    frame_id += 1

cap.release()
print(" Done scanning video.")
