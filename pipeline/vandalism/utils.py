import os
import json
import boto3
from urllib.parse import urlparse
import psycopg2

# Global constants
DB_SECRET_NAME = "ga8ed-db-userpass"
KV_SECRET_NAME = "kv_secrets"
DB_NAME = "postgres"
DB_SCHEMA = "ga8ed"
DB_TABLE = "video_metadata"


def get_db_userpass():
    client = boto3.client("secretsmanager")
    response = client.get_secret_value(SecretId=DB_SECRET_NAME)
    secret = json.loads(response["SecretString"])
    return secret


def get_gemini_api_key():
    client = boto3.client("secretsmanager")
    response = client.get_secret_value(SecretId=KV_SECRET_NAME)
    secret = json.loads(response["SecretString"])
    return secret["gemini_api_key"]


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
        SELECT video_path
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
