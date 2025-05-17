from fastapi import FastAPI, Query, HTTPException
from typing import Optional, List
from models import VideoResponse
from utils import execute_query, DB_SCHEMA, DB_TABLE

app = FastAPI(
    title="GA8ED Capstone Project: Real-time Incident Detection for Neighbourhood Safety"
)


@app.get("/videos", response_model=List[VideoResponse])
async def list_videos(
    limit: int = Query(10, ge=1, le=100),
    offset: int = Query(0, ge=0),
    has_alert: Optional[bool] = None,
):
    try:
        # Build the base query
        query = f"""
            SELECT *
            FROM {DB_SCHEMA}.{DB_TABLE}
            WHERE 1=1
        """
        params = []

        # Add alert filter
        if has_alert is not None:
            query += " AND (bear_alert = %s OR vandalism_alert = %s)"
            params.extend([has_alert, has_alert])

        query += f" ORDER BY updated_at desc"

        # Add pagination
        query += " LIMIT %s OFFSET %s"
        params.extend([limit, offset])

        results = execute_query(query, params)

        videos = []
        for row in results:
            video = VideoResponse(
                video_id=row[0],
                video_uri=row[1],
                tracked_predictions_uri=row[2],
                annotated_video_uri=row[3],
                stability_score=row[4],
                bear_alert=row[5],
                vandalism_genai_response=row[6],
                vandalism_alert=row[7],
                created_at=row[8],
                updated_at=row[9],
            )
            videos.append(video)

        return videos

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/videos/{video_id}", response_model=VideoResponse)
async def get_video(video_id: str):
    try:
        query = f"""
            SELECT *
            FROM {DB_SCHEMA}.{DB_TABLE}
            WHERE video_id = %s
        """

        results = execute_query(query, (video_id,))

        if not results:
            raise HTTPException(status_code=404, detail="Video not found")

        result = results[0]
        video = VideoResponse(
            video_id=result[0],
            video_uri=result[1],
            tracked_predictions_uri=result[2],
            annotated_video_uri=result[3],
            stability_score=result[4],
            bear_alert=result[5],
            vandalism_genai_response=result[6],
            vandalism_alert=result[7],
            created_at=result[8],
            updated_at=result[9],
        )

        return video

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
