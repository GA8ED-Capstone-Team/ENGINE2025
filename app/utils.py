import psycopg2
import boto3
import json

# Database configuration
DB_SECRET_NAME = "ga8ed-db-userpass"
DB_NAME = "postgres"
DB_SCHEMA = "ga8ed"
DB_TABLE = "video_metadata"


def get_db_connection():
    """Get a database connection using credentials from AWS Secrets Manager"""
    client = boto3.client("secretsmanager")
    response = client.get_secret_value(SecretId=DB_SECRET_NAME)
    secret = json.loads(response["SecretString"])

    return psycopg2.connect(
        dbname=DB_NAME,
        user=secret["username"],
        password=secret["password"],
        host=secret["host"],
        port=secret["port"],
    )


def execute_query(query, params=None):
    """Execute a database query and return results"""
    conn = None
    cur = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(query, params or ())
        return cur.fetchall()
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()
