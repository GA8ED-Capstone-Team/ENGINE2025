# Video Detection API

A FastAPI-based REST API for managing and retrieving video metadata from the GA8ED project. This API provides endpoints to list videos with filtering and pagination capabilities, and to retrieve detailed information about specific videos.

## Features

- List videos with filtering and pagination
- Get detailed information about specific videos
- Secure database connection using AWS Secrets Manager
- Dockerized for easy deployment

## API Endpoints

### List Videos
```
GET /videos
```

Query Parameters:
- `limit` (int, default=10): Number of records to return (1-100)
- `offset` (int, default=0): Number of records to skip
- `has_alert` (bool, optional): Filter videos that have either bear or vandalism alerts

Results are always sorted by `updated_at` in descending order (newest first).

### Get Video Details
```
GET /videos/{video_id}
```

Path Parameters:
- `video_id` (str): Unique identifier of the video

## Project Structure

```
app/
├── server.py      # FastAPI application and endpoints
├── models.py      # Data models using Pydantic
├── utils.py       # Utility functions for database operations
├── requirements.txt
└── Dockerfile
```

## Setup and Installation

### Prerequisites
- Python 3.11+
- Docker
- AWS credentials configured for Secrets Manager access
- PostgreSQL database

### Local Development

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
uvicorn server:app --reload
```

### Docker Deployment

1. Build the Docker image:
```bash
docker build -t video-detection-api .
```

2. Run the container:
```bash
docker run -p 8000:8000 video-detection-api
```

## Environment Variables

The application uses AWS Secrets Manager to securely manage database credentials. Make sure you have the following secret configured in AWS Secrets Manager:

- Secret Name: `ga8ed-db-userpass`
- Required fields:
  - username
  - password
  - host
  - port

## API Documentation

Once the application is running, you can access:
- Swagger UI documentation: `http://localhost:8000/docs`
- ReDoc documentation: `http://localhost:8000/redoc`

<!-- ## Database Schema

The application uses the following database schema:

```sql
CREATE TABLE ga8ed.video_metadata (
    video_id VARCHAR PRIMARY KEY,
    video_uri VARCHAR NOT NULL,
    tracked_predictions_uri VARCHAR NOT NULL,
    annotated_video_uri VARCHAR,
    stability_score FLOAT,
    bear_alert BOOLEAN,
    vandalism_genai_response TEXT,
    vandalism_alert BOOLEAN,
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);
``` -->