"""
Integration Tests for FastAPI Endpoints.
"""

import os
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from api.main import app, lifespan
from src.train import train_baseline_pipeline


@pytest.fixture(scope="module")
def api_client(tmp_path_factory):
    tmp_dir = tmp_path_factory.mktemp("api_artifacts")
    artifact_dir = os.path.join(tmp_dir, "artifacts")

    train_df = pd.DataFrame({
        "text": [
            "Severe flood warning issued",
            "Forest fire burning near highway",
            "Earthquake magnitude 6.5 hits region",
            "Eating lunch at a cafe",
            "Learning Python programming today",
            "Weekend getaway trip to the mountains"
        ],
        "target": [1, 1, 1, 0, 0, 0]
    })
    val_df = pd.DataFrame({
        "text": ["Flood disaster alert", "Nice weather today"],
        "target": [1, 0]
    })

    train_baseline_pipeline(train_df, val_df, artifact_dir=artifact_dir)
    os.environ["ARTIFACT_DIR"] = artifact_dir

    with TestClient(app) as client:
        yield client


def test_root_endpoint(api_client):
    response = api_client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "running"
    assert "docs_url" in data


def test_health_endpoint(api_client):
    response = api_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


def test_predict_single_endpoint(api_client):
    payload = {"text": "Wildfire spreading rapidly in Northern Texas #wildfire"}
    response = api_client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "label" in data
    assert data["label"] in ["Disaster", "Not Disaster"]
    assert "confidence" in data
    assert "latency_ms" in data
    assert data["latency_ms"] >= 0.0


def test_predict_batch_endpoint(api_client):
    payload = {
        "tweets": [
            "Severe flooding in coastal areas #disaster",
            "Had a great workout session!"
        ]
    }
    response = api_client.post("/predict/batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_tweets"] == 2
    assert len(data["predictions"]) == 2


def test_predict_invalid_input(api_client):
    response = api_client.post("/predict", json={"text": "   "})
    assert response.status_code == 400
