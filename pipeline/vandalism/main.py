import base64
import boto3
import google.generativeai as genai
import json
import os
import tempfile
from urllib.parse import urlparse

classes_to_check = {"car", "person"}
VANDALISM_PROMPT = """
"You are a neighborhood safety assistant. 
Does this video show someone damaging a car—scratching, kicking, or breaking it? 
If yes, respond with: 'Yes: [action]'. If not, say 'No: [reason]'."
"""

s3 = boto3.client("s3")


def download_video_from_s3(s3_uri: str) -> str:

    parsed = urlparse(s3_uri)
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")
    local_path = os.path.join(tempfile.gettempdir(), os.path.basename(key))
    s3.download_file(bucket, key, local_path)
    return local_path


def download_predictions_from_s3(s3_uri):

    parsed = urlparse(s3_uri)
    bucket, key = parsed.netloc, parsed.path.lstrip("/")
    local_path = os.path.join(tempfile.gettempdir(), os.path.basename(key))
    s3.download_file(bucket, key, local_path)
    return local_path, bucket, key


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


def analyze_video_with_gemini(video_path: str, gemini_api_key: str) -> None:

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
        print(f"\nGemini Response:\n{response.text}\n")
    except Exception as e:
        print(f"Gemini API error: {e}")


def main():
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    video_s3_uri = os.getenv("VIDEO_S3_PATH", "")
    tracked_predictions = os.getenv("TRACKED_PREDICTIONS", "")

    if not gemini_api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set.")

    if not tracked_predictions:
        raise ValueError("Detections is not provided.")

    if not contains_person_or_car(tracked_predictions):
        print("No person or car detections found. Skipping Gemini API call.")
        return

    video_path = download_video_from_s3(video_s3_uri)
    analyze_video_with_gemini(video_path, gemini_api_key)


if __name__ == "__main__":
    main()
