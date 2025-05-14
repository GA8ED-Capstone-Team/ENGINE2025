import psycopg2
from datetime import datetime
import boto3
import json

# Global constants
DB_SECRET_NAME = "ga8ed-db-userpass"
DB_NAME = "postgres"
DB_SCHEMA = "ga8ed"
DB_TABLE = "video_metadata"
TABLE_SCHEMA = "video_id, video_uri, tracked_predictions_uri, annotated_video_uri, stability_score, bear_alert, vandalism_genai_response, vandalism_alert, created_at, updated_at"


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
        record_dict["created_at"],
        record_dict["updated_at"],
    )

    cur.execute(
        f"""
        INSERT INTO {DB_SCHEMA}.{DB_TABLE} 
        ({TABLE_SCHEMA}) 
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """,
        record,
    )
    conn.commit()
    cur.close()
    conn.close()
    print(f"Record inserted for Video ID: {record_dict['video_id']}")
