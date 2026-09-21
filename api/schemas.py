"""
Pydantic API Schemas for Disaster Tweets Classification API.
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field


class TweetRequest(BaseModel):
    """Single tweet prediction request schema."""
    text: str = Field(
        ...,
        description="Raw tweet text to classify.",
        examples=["Wildfires spreading rapidly across Northern Texas #wildfire"]
    )


class BatchTweetRequest(BaseModel):
    """Batch tweet prediction request schema."""
    tweets: List[str] = Field(
        ...,
        description="List of raw tweet strings to classify.",
        examples=[
            ["Wildfires spreading rapidly across Northern Texas #wildfire"],
            ["Just had a delicious coffee break at the central cafe!"]
        ]
    )


class ClassProbabilities(BaseModel):
    """Probability scores for binary target classes."""
    not_disaster: float = Field(..., alias="Not Disaster")
    disaster: float = Field(..., alias="Disaster")

    model_config = ConfigDict(populate_by_name=True)



class PredictionResponse(BaseModel):
    """Single tweet classification response schema."""
    text: str
    cleaned_text: str
    label: str = Field(..., description="Predicted class label ('Disaster' or 'Not Disaster')")
    label_id: int = Field(..., description="Numeric class label (1 or 0)")
    confidence: float = Field(..., description="Confidence score between 0.0 and 1.0")
    probabilities: Dict[str, float]
    latency_ms: Optional[float] = Field(None, description="Inference latency in milliseconds")


class BatchPredictionResponse(BaseModel):
    """Batch tweet classification response schema."""
    predictions: List[PredictionResponse]
    total_tweets: int
    latency_ms: float


class HealthResponse(BaseModel):
    """API Health Check response schema."""
    status: str
    model_loaded: bool
    artifact_dir: str
    version: str = "1.0.0"
