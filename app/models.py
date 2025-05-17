from sqlalchemy import Column, String, Float, Boolean, DateTime, Text
from database import Base
from pydantic import BaseModel
from typing import Optional
from datetime import datetime


class VideoMetadata(Base):
    __tablename__ = "video_metadata"
    __table_args__ = {"schema": "ga8ed"}

    video_id = Column(String, primary_key=True)
    video_uri = Column(String)
    tracked_predictions_uri = Column(String)
    annotated_video_uri = Column(String, nullable=True)
    stability_score = Column(Float, nullable=True)
    bear_alert = Column(Boolean, nullable=True)
    vandalism_genai_response = Column(Text, nullable=True)
    vandalism_alert = Column(Boolean, nullable=True)
    created_at = Column(DateTime)
    updated_at = Column(DateTime)


class VideoResponse(BaseModel):
    video_id: str
    video_uri: str
    tracked_predictions_uri: str
    annotated_video_uri: str
    stability_score: Optional[float]
    bear_alert: Optional[bool]
    vandalism_genai_response: Optional[str]
    vandalism_alert: Optional[bool]
    created_at: datetime
    updated_at: datetime
