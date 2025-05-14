import os
import cv2
import boto3
import json
import tempfile
import time
from urllib.parse import urlparse
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import uuid
from datetime import datetime
from utils import insert_video_record

s3_path = os.environ.get("VIDEO_S3_PATH")
if not s3_path:
    raise ValueError("video path not provided")

# TODO: Replace with custom model
model = YOLO("yolov8n.pt")
s3 = boto3.client("s3")


def download_video_from_s3(s3_uri):
    parsed = urlparse(s3_uri)
    bucket, key = parsed.netloc, parsed.path.lstrip("/")
    local_path = os.path.join(tempfile.gettempdir(), os.path.basename(key))
    s3.download_file(bucket, key, local_path)
    return local_path, bucket, key


def upload_json_to_s3(json_data, bucket, prefix, filename="tracked_predictions.json"):
    output_path = os.path.join(tempfile.gettempdir(), filename)
    with open(output_path, "w") as f:
        json.dump(json_data, f, indent=2)
    s3.upload_file(output_path, bucket, f"{prefix}/{filename}")
    print(f"✅ Uploaded results to s3://{bucket}/{prefix}/{filename}")


def run_yolo_deepsort(video_path):

    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    frame_data = []

    fps = cap.get(cv2.CAP_PROP_FPS)
    skip = max(1, int(fps))
    tracker = DeepSort(
        max_age=skip, nn_budget=70, max_iou_distance=0.5
    )  # could be overkill, check it out

    while True:
        ret, frame = cap.read()
        if frame_idx % skip != 0:
            frame_idx += 1
            continue

        if not ret:
            break

        results = model(frame)[0]
        detections = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append(([x1, y1, x2 - x1, y2 - y1], conf, cls_id))

        tracks = tracker.update_tracks(detections, frame=frame)

        tracked_objects = []
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            ltrb, cls_id, conf = track.to_ltrb(), track.det_class, track.det_conf
            class_name = (
                model.names[cls_id] if cls_id < len(model.names) else str(cls_id)
            )
            x1, y1, x2, y2 = ltrb
            tracked_objects.append(
                {
                    "track_id": int(track_id),
                    "class_id": int(cls_id),
                    "class_name": class_name,
                    "confidence": round(conf, 4) if conf is not None else None,
                    "bbox": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                }
            )

        frame_data.append({"frame_number": frame_idx, "tracks": tracked_objects})
        frame_idx += 1
        print(f"Frame idx: {frame_idx}")

    return {"video": os.path.basename(video_path), "frames": frame_data}


if __name__ == "__main__":
    start_time = time.time()
    video_id = str(uuid.uuid4())
    print(f"Video ID: {video_id}")
    print(f"Downloading video from {s3_path}")
    local_video, bucket, key = download_video_from_s3(s3_path)

    print("Running YOLO + DeepSORT inference...")
    predictions = run_yolo_deepsort(local_video)
    predictions["video_id"] = video_id

    output_prefix = (
        f"object_detection_results/{os.path.splitext(os.path.basename(key))[0]}"
    )
    print("Uploading tracked results to S3...")
    upload_json_to_s3(predictions, bucket, output_prefix)

    # Insert record into PostgreSQL
    tracked_predictions_path = f"s3://{bucket}/{output_prefix}/tracked_predictions.json"
    created_at = datetime.now()
    updated_at = created_at
    record_dict = {
        "video_id": video_id,
        "tracked_predictions_path": tracked_predictions_path,
        "created_at": created_at,
        "updated_at": updated_at
    }
    insert_video_record(record_dict)

    print(f"Processing complete. Total time taken: {time.time() - start_time}")
