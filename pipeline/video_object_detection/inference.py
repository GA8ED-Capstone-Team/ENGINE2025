import os
import tempfile
import time
import uuid
from datetime import datetime
from utils import (
    download_video_from_s3,
    run_yolo_deepsort,
    upload_json_to_s3,
    create_annotated_video,
    upload_video_to_s3,
    insert_video_record,
)


def main():
    start_time = time.time()
    video_id = str(uuid.uuid4())
    print(f"Video ID: {video_id}")

    s3_path = os.environ.get("VIDEO_S3_PATH")
    if not s3_path:
        raise ValueError("video path not provided")

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
    print("Creating annotated video")
    input_format = os.path.splitext(local_video)[1].lower()
    annotated_video_path = os.path.join(
        tempfile.gettempdir(), f"annotated_{os.path.basename(local_video)}"
    )
    create_annotated_video(local_video, predictions["frames"], annotated_video_path)
    upload_video_to_s3(
        annotated_video_path,
        bucket,
        output_prefix,
        filename=f"annotated_video{input_format}",
    )

    # Insert record into PostgreSQL
    tracked_predictions_uri = f"s3://{bucket}/{output_prefix}/tracked_predictions.json"
    annotated_video_uri = f"s3://{bucket}/{output_prefix}/annotated_video{input_format}"
    t = datetime.now()
    record_dict = {
        "video_id": video_id,
        "video_uri": s3_path,
        "tracked_predictions_uri": tracked_predictions_uri,
        "annotated_video_uri": annotated_video_uri,
        "created_at": t,
        "updated_at": t,
    }
    print(f"Inserting record for {video_id} into db")
    insert_video_record(record_dict)

    print(f"Processing complete. Total time taken: {time.time() - start_time}")


if __name__ == "__main__":
    main()
