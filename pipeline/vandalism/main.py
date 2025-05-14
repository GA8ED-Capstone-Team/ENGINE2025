import base64
import boto3
import google.generativeai as genai
import json
import os
import tempfile
from urllib.parse import urlparse

from utils import (
    get_video_id_from_s3_path,
    update_video_record,
    parse_vandalism_response,
    get_gemini_api_key,
    get_video_path_from_db,
)

classes_to_check = {"person", "car", "bus", "truck", "motorcycle", "bicycle", "train"}
VANDALISM_PROMPT = """
"You are a neighborhood safety assistant. 
Does this video show someone damaging a car—scratching, kicking, or breaking it? 
If yes, respond with: 'Yes: [action]'. If not, say 'No: [reason]'."
"""

s3 = boto3.client("s3")


def download_predictions_from_s3(s3_uri):
    parsed = urlparse(s3_uri)
    bucket, key = parsed.netloc, parsed.path.lstrip("/")
    local_path = os.path.join(tempfile.gettempdir(), os.path.basename(key))
    s3.download_file(bucket, key, local_path)
    return local_path, bucket, key


def download_video_from_s3(s3_uri: str) -> str:
    parsed = urlparse(s3_uri)
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")
    local_path = os.path.join(tempfile.gettempdir(), os.path.basename(key))
    s3.download_file(bucket, key, local_path)
    return local_path


def contains_person_or_car(detections_path: str) -> bool:
    detections_path_local, _, _ = download_predictions_from_s3(detections_path)
    with open(detections_path_local, "r") as f:
        data = json.load(f)

    for frame in data.get("frames", []):
        for track in frame.get("tracks", []):
            class_name = track.get("class_name", "").lower()
            if class_name in classes_to_check:
                return True
    return False


def analyze_video_with_gemini(video_path: str, gemini_api_key: str) -> tuple[bool, str]:
    genai.configure(api_key=gemini_api_key)
    gemini_model = genai.GenerativeModel("gemini-1.5-flash")

    with open(video_path, "rb") as f:
        video_data = f.read()
        video_b64 = base64.b64encode(video_data).decode("utf-8")

    prompt = [
        {"inline_data": {"mime_type": "video/mp4", "data": video_b64}},
        VANDALISM_PROMPT,
    ]

    try:
        response = gemini_model.generate_content(prompt)
        response_text = response.text
        print(f"\nGemini Response:\n{response_text}\n")
        return parse_vandalism_response(response_text)
    except Exception as e:
        print(f"Gemini API error: {e}")
        return False, ""


def main():
    tracked_predictions = os.getenv("TRACKED_PREDICTIONS", "")

    # Get video_id from S3 path
    video_id = get_video_id_from_s3_path(tracked_predictions)

    # Custom logic to only prompt VLM if there are person or car detections (can be refined)
    if not contains_person_or_car(tracked_predictions):
        print("No person or car detections found. Skipping Gemini API call.")
        # Update database with no vandalism
        update_video_record(video_id, "No person or car detected", False)
        return

    # Get video path from database
    video_s3_uri = get_video_path_from_db(video_id)
    video_path = download_video_from_s3(video_s3_uri)
    gemini_api_key = get_gemini_api_key()
    vandalism_alert, vandalism_response = analyze_video_with_gemini(
        video_path, gemini_api_key
    )

    # Update database with vandalism analysis results
    update_video_record(video_id, vandalism_response, vandalism_alert)


if __name__ == "__main__":
    main()
