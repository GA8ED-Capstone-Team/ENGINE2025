import os
import json
import boto3
import tempfile
import base64
import google.generativeai as genai
from urllib.parse import urlparse
import psycopg2

# Global constants
DB_SECRET_NAME = "ga8ed-db-userpass"
KV_SECRET_NAME = "ga8ed-secrets"
DB_NAME = "postgres"
DB_SCHEMA = "ga8ed"
DB_TABLE = "video_metadata"

# Vandalism detection configurations
CLASSES_TO_CHECK = {"person", "car", "bus", "truck", "motorcycle", "bicycle", "train"}
VANDALISM_PROMPT = """
"You are a neighborhood safety assistant. 
Does this video show someone damaging a car—scratching, kicking, or breaking it? 
If yes, respond with: 'Yes: [action]'. If not, say 'No: [reason]'."
"""


def get_db_userpass():
    client = boto3.client("secretsmanager")
    response = client.get_secret_value(SecretId=DB_SECRET_NAME)
    secret = json.loads(response["SecretString"])
    return secret


def get_gemini_api_key():
    client = boto3.client("secretsmanager")
    response = client.get_secret_value(SecretId=KV_SECRET_NAME)
    secret = json.loads(response["SecretString"])
    return secret["GEMINI_API_KEY"]


def get_video_id_from_s3_path(s3_path):
    parsed = urlparse(s3_path)
    path_parts = parsed.path.lstrip("/").split("/")
    # The video_id is the second-to-last part of the path
    return path_parts[-2]


def get_video_path_from_db(video_id):
    secrets = get_db_userpass()
    conn = psycopg2.connect(
        dbname=DB_NAME,
        user=secrets["username"],
        password=secrets["password"],
        host=secrets["host"],
        port=secrets["port"],
    )
    cur = conn.cursor()

    cur.execute(
        f"""
        SELECT video_uri
        FROM {DB_SCHEMA}.{DB_TABLE}
        WHERE video_id = %s
        """,
        (video_id,),
    )
    result = cur.fetchone()
    cur.close()
    conn.close()

    if result is None:
        raise ValueError(f"No video path found for video_id: {video_id}")
    return result[0]


def update_video_record(video_id, vandalism_genai_response, vandalism_alert):
    secrets = get_db_userpass()
    conn = psycopg2.connect(
        dbname=DB_NAME,
        user=secrets["username"],
        password=secrets["password"],
        host=secrets["host"],
        port=secrets["port"],
    )
    cur = conn.cursor()

    cur.execute(
        f"""
        UPDATE {DB_SCHEMA}.{DB_TABLE}
        SET vandalism_genai_response = %s,
            vandalism_alert = %s,
            updated_at = CURRENT_TIMESTAMP
        WHERE video_id = %s
        """,
        (vandalism_genai_response, vandalism_alert, video_id),
    )
    conn.commit()
    cur.close()
    conn.close()
    print(f"Updated record for Video ID: {video_id}")


def parse_vandalism_response(response_text):
    """
    Parse the Gemini response to determine if it indicates vandalism.
    Returns a tuple of (is_vandalism, response_text)
    """
    if not response_text:
        return False, ""
    response_lower = response_text.lower()
    is_vandalism = response_lower.startswith("yes")

    return is_vandalism, response_text


def download_predictions_from_s3(s3_uri):
    parsed = urlparse(s3_uri)
    bucket, key = parsed.netloc, parsed.path.lstrip("/")
    local_path = os.path.join(tempfile.gettempdir(), os.path.basename(key))
    s3 = boto3.client("s3")
    s3.download_file(bucket, key, local_path)
    return local_path, bucket, key


def download_video_from_s3(s3_uri: str) -> str:
    parsed = urlparse(s3_uri)
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")
    local_path = os.path.join(tempfile.gettempdir(), os.path.basename(key))
    s3 = boto3.client("s3")
    s3.download_file(bucket, key, local_path)
    return local_path


def contains_person_or_car(detections_path: str) -> bool:
    detections_path_local, _, _ = download_predictions_from_s3(detections_path)
    with open(detections_path_local, "r") as f:
        data = json.load(f)

    for frame in data.get("frames", []):
        for track in frame.get("tracks", []):
            class_name = track.get("class_name", "").lower()
            if class_name in CLASSES_TO_CHECK:
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
