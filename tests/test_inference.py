"""
Unit Tests for DisasterTweetPredictor Inference Engine.
"""

import os
import pandas as pd
import pytest
from src.inference import DisasterTweetPredictor
from src.train import train_baseline_pipeline


@pytest.fixture(scope="module")
def setup_artifacts(tmp_path_factory):
    tmp_dir = tmp_path_factory.mktemp("inference_artifacts")
    artifact_dir = os.path.join(tmp_dir, "artifacts")

    train_df = pd.DataFrame({
        "text": [
            "Emergency earthquake evacuation in city center",
            "Huge wildfire raging across the hills",
            "Severe flood warning issued by authorities",
            "Enjoying a hot cup of coffee this morning",
            "Walking my dog in the sunshine",
            "Had a great workout session at the gym"
        ],
        "target": [1, 1, 1, 0, 0, 0]
    })
    val_df = pd.DataFrame({
        "text": ["Flooding in streets", "Good morning world"],
        "target": [1, 0]
    })

    train_baseline_pipeline(train_df, val_df, artifact_dir=artifact_dir)
    return artifact_dir


def test_predictor_single(setup_artifacts):
    predictor = DisasterTweetPredictor(artifact_dir=setup_artifacts)

    res = predictor.predict("Wildfire spreading rapidly! #wildfire")
    assert "label" in res
    assert res["label"] in ["Disaster", "Not Disaster"]
    assert "confidence" in res
    assert "probabilities" in res
    assert "Disaster" in res["probabilities"]
    assert "Not Disaster" in res["probabilities"]
    assert res["cleaned_text"] == "wildfire spreading rapidly! wildfire"


def test_predictor_batch(setup_artifacts):
    predictor = DisasterTweetPredictor(artifact_dir=setup_artifacts)
    tweets = [
        "Major earthquake recorded near faultline",
        "Just bought a new book to read"
    ]
    results = predictor.predict_batch(tweets)
    assert len(results) == 2
    assert results[0]["label_id"] in [0, 1]
    assert results[1]["label_id"] in [0, 1]


def test_predictor_empty_input(setup_artifacts):
    predictor = DisasterTweetPredictor(artifact_dir=setup_artifacts)
    res = predictor.predict("")
    assert res["label"] == "Not Disaster"
    assert res["confidence"] == 0.5
