"""
FastAPI Application for Disaster Tweets Classification.

Exposes RESTful endpoints for health checks, single tweet prediction,
and batch tweet predictions. Loads model artifacts once during startup.
"""

import os
import sys
import time
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

# Ensure parent directory is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from api.schemas import (
    BatchPredictionResponse,
    BatchTweetRequest,
    HealthResponse,
    PredictionResponse,
    TweetRequest,
)
from src.inference import DisasterTweetPredictor

# Global Predictor Instance loaded once on lifespan startup
predictor: Optional[DisasterTweetPredictor] = None
ARTIFACT_DIR = os.getenv("ARTIFACT_DIR", "artifacts")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI Lifespan event handler.
    Ensures model weights are loaded ONCE at application startup rather than per request.
    """
    global predictor
    print("[API Startup] Initializing DisasterTweetPredictor model weights...")
    try:
        predictor = DisasterTweetPredictor(artifact_dir=ARTIFACT_DIR)
        print("[API Startup] Model loaded successfully into API runtime memory.")
    except Exception as e:
        print(f"[API Startup ERROR] Failed to load model from '{ARTIFACT_DIR}': {e}")
        predictor = None

    yield

    print("[API Shutdown] Releasing resources...")
    predictor = None


app = FastAPI(
    title="Disaster Tweets Classification API",
    description="Production-ready REST API to predict whether social media tweets describe real-world disasters or everyday events.",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for frontend/integration support
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["System"])
async def root():
    """Root endpoint providing API information and documentation link."""
    return {
        "title": "Disaster Tweets Classification API",
        "status": "running",
        "docs_url": "/docs",
        "redoc_url": "/redoc",
        "health_check": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Health check endpoint evaluating API readiness and model load status."""
    is_loaded = predictor is not None and predictor.model is not None
    return HealthResponse(
        status="healthy" if is_loaded else "degraded",
        model_loaded=is_loaded,
        artifact_dir=ARTIFACT_DIR
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict_tweet(request: TweetRequest):
    """
    Classify a single raw tweet string.

    Returns predicted label ('Disaster' vs 'Not Disaster'), confidence score,
    class probabilities, and execution latency.
    """
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded. Ensure training artifacts exist in the artifacts directory."
        )

    if not request.text or not request.text.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Input text string cannot be empty."
        )

    start_time = time.perf_counter()
    result = predictor.predict(request.text)
    latency_ms = (time.perf_counter() - start_time) * 1000.0

    return PredictionResponse(
        text=result["text"],
        cleaned_text=result["cleaned_text"],
        label=result["label"],
        label_id=result["label_id"],
        confidence=result["confidence"],
        probabilities=result["probabilities"],
        latency_ms=round(latency_ms, 2)
    )


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Inference"])
async def predict_tweet_batch(request: BatchTweetRequest):
    """
    Classify a batch list of raw tweet strings in a single call.
    """
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded. Ensure training artifacts exist in the artifacts directory."
        )

    if not request.tweets:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Tweets list cannot be empty."
        )

    start_time = time.perf_counter()
    batch_results = predictor.predict_batch(request.tweets)
    total_latency_ms = (time.perf_counter() - start_time) * 1000.0

    responses = [
        PredictionResponse(
            text=res["text"],
            cleaned_text=res["cleaned_text"],
            label=res["label"],
            label_id=res["label_id"],
            confidence=res["confidence"],
            probabilities=res["probabilities"],
            latency_ms=None
        )
        for res in batch_results
    ]

    return BatchPredictionResponse(
        predictions=responses,
        total_tweets=len(responses),
        latency_ms=round(total_latency_ms, 2)
    )
