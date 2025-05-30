import psycopg2
from datetime import datetime
import boto3
import json
import os
import cv2
import tempfile
from urllib.parse import urlparse
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# YOLO and DeepSORT configurations
# TODO: Replace this with fine-tuned model
model = YOLO("yolov8m.pt")
tracker = DeepSort(
    max_age=2,
    nn_budget=30,
    max_iou_distance=0.5,
)

# Global constants
DB_SECRET_NAME = "ga8ed-db-userpass"
DB_NAME = "postgres"
DB_SCHEMA = "ga8ed"
DB_TABLE = "video_metadata_2"
TABLE_SCHEMA = "video_id, video_uri, tracked_predictions_uri, annotated_video_uri, stability_score, bear_alert, max_speed, speed_alert, vandalism_genai_response, vandalism_alert, created_at, updated_at"

# Colors for different object groups (BGR format)
COLORS = {
    "person": (0, 255, 0),
    "vehicle": (255, 0, 0),
    "animal": (0, 0, 255),
    "default": (255, 255, 0),
}

# Video codec mapping
VIDEO_CODECS = {
    ".mp4": "mp4v",
    ".avi": "XVID",
    ".mov": "avc1",
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


def get_db_userpass():
    client = boto3.client("secretsmanager")
    response = client.get_secret_value(SecretId=DB_SECRET_NAME)
    secret = json.loads(response["SecretString"])
    return secret


def insert_video_record(record_dict):
    secrets = get_db_userpass()
    conn = psycopg2.connect(
        dbname=DB_NAME,
        user=secrets["username"],
        password=secrets["password"],
        host=secrets["host"],
        port=secrets["port"],
    )
    cur = conn.cursor()
    record = (
        record_dict["video_id"],
        record_dict["video_uri"],
        record_dict["tracked_predictions_uri"],
        record_dict["annotated_video_uri"],
        None,
        None,
        None,
        None,
        None,
        None,
        record_dict["created_at"],
        record_dict["updated_at"],
    )

    cur.execute(
        f"""
        INSERT INTO {DB_SCHEMA}.{DB_TABLE} 
        ({TABLE_SCHEMA}) 
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """,
        record,
    )
    conn.commit()
    cur.close()
    conn.close()
    print(f"Record inserted for Video ID: {record_dict['video_id']}")


def get_object_group(class_name):
    """Get the group for an object class"""
    return OBJECT_GROUPS.get(class_name.lower(), "default")


def draw_detections(frame, tracked_objects):
    """
    Draw bounding boxes and labels on the frame
    """
    for obj in tracked_objects:
        if obj["confidence"] is None:
            continue
        x1, y1, x2, y2 = [int(coord) for coord in obj["bbox"]]
        class_name = obj["class_name"]
        conf = obj["confidence"]
        group = get_object_group(class_name)
        color = COLORS[group]

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{class_name} - ({conf:.2f})"
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

    input_format = os.path.splitext(video_path)[1].lower()
    if input_format not in VIDEO_CODECS:
        print(f"Warning: Unknown video format {input_format}, defaulting to mp4v codec")
        output_path = os.path.splitext(output_path)[0] + ".mp4"
    codec = VIDEO_CODECS.get(input_format, "mp4v")

    fourcc = cv2.VideoWriter_fourcc(*codec)
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
    s3 = boto3.client("s3")
    s3.download_file(bucket, key, local_path)
    return local_path, bucket, key


def upload_json_to_s3(json_data, bucket, prefix, filename="tracked_predictions.json"):
    output_path = os.path.join(tempfile.gettempdir(), filename)
    with open(output_path, "w") as f:
        json.dump(json_data, f, indent=2)
    s3 = boto3.client("s3")
    s3.upload_file(output_path, bucket, f"{prefix}/{filename}")
    print(f"✅ Uploaded results to s3://{bucket}/{prefix}/{filename}")


def upload_video_to_s3(video_path, bucket, prefix, filename="annotated_video.mp4"):
    s3 = boto3.client("s3")
    s3.upload_file(video_path, bucket, f"{prefix}/{filename}")
    print(f"✅ Uploaded annotated video to s3://{bucket}/{prefix}/{filename}")


def run_yolo_deepsort(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    frame_data = []

    fps = cap.get(cv2.CAP_PROP_FPS)
    skip = max(1, int(fps) // 2 * 5)

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
