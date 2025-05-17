import psycopg2
import boto3
import json
import os
from logger import log_info, log_error, log_debug

# Set AWS region
os.environ["AWS_DEFAULT_REGION"] = "us-east-1"

# Database configuration
DB_SECRET_NAME = "ga8ed-db-userpass"
DB_NAME = "postgres"
DB_SCHEMA = "ga8ed"
DB_TABLE = "video_metadata"


def get_db_connection():
    """Get a database connection using credentials from AWS Secrets Manager"""
    try:
        log_debug(
            f"Getting database credentials from Secrets Manager: {DB_SECRET_NAME}"
        )
        client = boto3.client("secretsmanager")
        response = client.get_secret_value(SecretId=DB_SECRET_NAME)
        secret = json.loads(response["SecretString"])
        log_debug("Successfully retrieved database credentials")

        log_debug(
            f"Connecting to database: {DB_NAME} at {secret['host']}:{secret['port']}"
        )
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=secret["username"],
            password=secret["password"],
            host=secret["host"],
            port=secret["port"],
        )
        log_info("Successfully connected to database")
        return conn
    except Exception as e:
        log_error(e, "Error getting database connection")
        raise


def execute_query(query, params=None):
    """Execute a database query and return results"""
    conn = None
    cur = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        log_debug(f"Executing query: {query} with params: {params}")
        cur.execute(query, params or ())
        results = cur.fetchall()
        log_debug(f"Query returned {len(results)} results")
        return results
    except Exception as e:
        log_error(e, "Error executing database query")
        raise
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()
            log_debug("Database connection closed")
