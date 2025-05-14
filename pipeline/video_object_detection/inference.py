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
model = YOLO("yolov8m.pt")
s3 = boto3.client("s3")

# Colors for different object groups (BGR format)
COLORS = {
    "person": (0, 255, 0),  # Green for people
    "vehicle": (255, 0, 0),  # Blue for vehicles
    "animal": (0, 0, 255),  # Red for animals
    "default": (255, 255, 0),  # Cyan for everything else
}

# Object class to group mapping
OBJECT_GROUPS = {
    # People
    "person": "person",
    # Vehicles
    "car": "vehicle",
    "truck": "vehicle",
    "bus": "vehicle",
    "motorcycle": "vehicle",
    "bicycle": "vehicle",
    "train": "vehicle",
    "boat": "vehicle",
    "airplane": "vehicle",
    # Animals (only those that YOLO can detect)
    "bear": "animal",
    "bird": "animal",
    "cat": "animal",
    "cow": "animal",
    "dog": "animal",
    "elephant": "animal",
    "giraffe": "animal",
    "horse": "animal",
    "sheep": "animal",
    "zebra": "animal",
}


def get_object_group(class_name):
    """Get the group for an object class"""
    return OBJECT_GROUPS.get(class_name.lower(), "default")


def draw_detections(frame, tracked_objects):
    """
    Draw bounding boxes and labels on the frame
    """
    for obj in tracked_objects:
        x1, y1, x2, y2 = [int(coord) for coord in obj["bbox"]]
        class_name = obj["class_name"]
        conf = obj["confidence"]
        group = get_object_group(class_name)
        color = COLORS[group]

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{class_name} - ({conf:.2f})" if conf else f"{class_name}"
        (label_width, label_height), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
        )
        cv2.rectangle(
            frame, (x1, y1 - label_height - 10), (x1 + label_width, y1), color, -1
        )

        cv2.putText(
            frame,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
        )

    return frame


def create_annotated_video(video_path, frame_data, output_path):
    """Create a summary video showing only frames with detections at a lower FPS"""
    cap = cv2.VideoCapture(video_path)
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Calculate the new FPS based on frame skip
    # If we processed every N-th frame, new FPS should be original_fps/N
    frame_numbers = [frame["frame_number"] for frame in frame_data]
    if len(frame_numbers) > 1:
        frame_skip = frame_numbers[1] - frame_numbers[0]
        new_fps = original_fps / frame_skip
    else:
        new_fps = original_fps
    print(f"Original FPS: {original_fps}, New FPS: {new_fps}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, new_fps, (width, height))
    frame_detections = {frame["frame_number"]: frame["tracks"] for frame in frame_data}
    for frame_idx in sorted(frame_detections.keys()):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
        frame = draw_detections(frame, frame_detections[frame_idx])
        out.write(frame)
    cap.release()
    out.release()

    return output_path


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


def upload_video_to_s3(video_path, bucket, prefix, filename="annotated_video.mp4"):
    s3.upload_file(video_path, bucket, f"{prefix}/{filename}")
    print(f"✅ Uploaded annotated video to s3://{bucket}/{prefix}/{filename}")


def run_yolo_deepsort(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    frame_data = []

    fps = cap.get(cv2.CAP_PROP_FPS)
    skip = max(1, int(fps) // 2)  # Process every 30th frame at 30fps
    tracker = DeepSort(
        max_age=2,  # Track dies after missing 2 detection frames
        nn_budget=30,  # Reduced from 70 to 30 for speed
        max_iou_distance=0.5,
    )

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

    output_prefix = f"object_detection_results/{video_id}"
    print("Uploading tracked results to S3...")
    upload_json_to_s3(
        predictions, bucket, output_prefix, filename="tracked_predictions.json"
    )

    # Create and upload annotated video
    print("Creating annotated video...")
    annotated_video_path = os.path.join(
        tempfile.gettempdir(), f"annotated_{os.path.basename(local_video)}"
    )
    create_annotated_video(local_video, predictions["frames"], annotated_video_path)
    upload_video_to_s3(
        annotated_video_path, bucket, output_prefix, filename="annotated_video.mp4"
    )

    # Insert record into PostgreSQL
    tracked_predictions_uri = f"s3://{bucket}/{output_prefix}/tracked_predictions.json"
    annotated_video_uri = f"s3://{bucket}/{output_prefix}/annotated_video.mp4"
    t = datetime.now()
    record_dict = {
        "video_id": video_id,
        "video_uri": s3_path,
        "tracked_predictions_uri": tracked_predictions_uri,
        "annotated_video_uri": annotated_video_uri,
        "created_at": t,
        "updated_at": t,
    }
    insert_video_record(record_dict)

    print(f"Processing complete. Total time taken: {time.time() - start_time}")
