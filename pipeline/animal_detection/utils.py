import os
import json
import numpy as np
import psycopg2
import boto3
import json
from urllib.parse import urlparse

# Global constants
DB_SECRET_NAME = "ga8ed-db-userpass"
DB_NAME = "postgres"
DB_SCHEMA = "ga8ed"
DB_TABLE = "video_metadata"

BEAR_CLASS = 21


def get_db_userpass():
    client = boto3.client("secretsmanager")
    response = client.get_secret_value(SecretId=DB_SECRET_NAME)
    secret = json.loads(response["SecretString"])
    return secret


def get_video_id_from_s3_path(s3_path):
    parsed = urlparse(s3_path)
    path_parts = parsed.path.lstrip("/").split("/")
    # The video_id is the second-to-last part of the path
    return path_parts[-2]


def update_video_record(video_id, stability_score, bear_alert):
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
        SET stability_score = %s,
            bear_alert = %s,
            updated_at = CURRENT_TIMESTAMP
        WHERE video_id = %s
        """,
        (stability_score, bear_alert, video_id),
    )
    conn.commit()
    cur.close()
    conn.close()
    print(f"Updated record for Video ID: {video_id}")


def compute_stability_scores(
    json_input_path: str,
    json_output_path: str,
    stability_threshold: float,
    wild_animals: dict,
    min_frame_persistence: int = 10,
    window_size: int = 100,
    raise_alerts: bool = True,
):
    """
    Reads tracked_predictions.json, computes stability scores over
    the last `window_size` frames, writes them to json_output_path,
    and optionally prints alerts.
    """
    # Load tracked_predictions.json
    with open(json_input_path, "r") as f:
        data = json.load(f)
    frames = data.get("frames", [])
    window = frames[-window_size:] if len(frames) > window_size else frames
    total_tracked = len(window)

    # Accumulate confidences & persistence
    object_confidences = {}
    object_persistence = {}
    for frame in window:
        seen = set()
        for tr in frame.get("tracks", []):
            cls, conf = tr["class_id"], tr.get("confidence")
            if conf is not None:
                object_confidences.setdefault(cls, []).append(conf)
            if cls not in seen:
                object_persistence[cls] = object_persistence.get(cls, 0) + 1
                seen.add(cls)

    # Compute stability scores and write to s3
    stability_scores = {}
    animal_alerts = {}  # Dictionary to store alerts for each wild animal
    for cls, confs in object_confidences.items():
        pers = object_persistence.get(cls, 0)
        if pers > 0:
            score = (np.mean(confs) * pers) / total_tracked
            stability_scores[cls] = {
                "stability_score": score,
                "mean_confidence": float(np.mean(confs)),
                "persistence": pers,
            }

            # Check if this is a wild animal and if it should trigger an alert
            if cls in wild_animals:
                is_bear = cls == BEAR_CLASS
                should_alert = (score > stability_threshold) or (
                    is_bear
                    and pers >= min_frame_persistence
                    and float(np.mean(confs)) > 0.5
                )
                animal_alerts[wild_animals[cls]] = should_alert

                if should_alert and raise_alerts:
                    name = wild_animals[cls].upper()
                    print(f"ALERT: {name} detected (score={score:.3f})")

    with open(json_output_path, "w") as f:
        json.dump(
            {
                "video": data.get("video"),
                "total_tracked_frames": total_tracked,
                "scores": stability_scores,
            },
            f,
            indent=2,
        )
    print(f"Stability scores written to {json_output_path}")

    return animal_alerts, stability_scores
