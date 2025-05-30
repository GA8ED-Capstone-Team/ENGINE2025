from pydantic import BaseModel
from typing import Optional
from datetime import datetime


class VideoResponse(BaseModel):
    video_id: str
    video_uri: str
    tracked_predictions_uri: str
    annotated_video_uri: str
    stability_score: Optional[float]
    bear_alert: Optional[bool]
    max_speed: Optional[float]
    speed_alert: Optional[bool]
    vandalism_genai_response: Optional[str]
    vandalism_alert: Optional[bool]
    created_at: datetime
    updated_at: datetime
