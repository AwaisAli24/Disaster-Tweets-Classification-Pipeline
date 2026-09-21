"""
Integration Tests for Model Training Pipeline.
"""

import json
import os
import pandas as pd
import pytest
from data.prepare_data import prepare_dataset
from src.train import run_training, train_baseline_pipeline


def test_prepare_dataset(tmp_path):
    data_dir = str(tmp_path)
    csv_path = os.path.join(data_dir, "DisasterTweets.csv")

    df_dummy = pd.DataFrame({
        "Tweets": [
            "Severe flood warning issued for coast",
            "Having a lovely picnic at the park",
            "Wildfire destroying forest trees",
            "Reading a good book at home"
        ],
        "Disaster": ["Floods", None, "Wildfire", None]
    })
    df_dummy.to_csv(csv_path, index=False)

    train_path, test_path = prepare_dataset(data_dir=data_dir, test_size=0.5, random_state=42)
    assert os.path.exists(train_path)
    assert os.path.exists(test_path)

    df_train = pd.read_csv(train_path)
    assert "text" in df_train.columns
    assert "target" in df_train.columns
    assert len(df_train) == 2


def test_train_baseline_pipeline(tmp_path):
    artifact_dir = os.path.join(tmp_path, "artifacts")
    train_df = pd.DataFrame({
        "text": [
            "Heavy flooding reported in downtown",
            "Major wildfire spreading fast",
            "Emergency earthquake evacuation alert",
            "Baking chocolate chip cookies today",
            "Watching movies with friends on weekend",
            "Beautiful sunset at the beach"
        ],
        "target": [1, 1, 1, 0, 0, 0]
    })
    val_df = pd.DataFrame({
        "text": [
            "Disaster alert flood warning",
            "Relaxing afternoon drinking tea"
        ],
        "target": [1, 0]
    })

    pipeline, metrics = train_baseline_pipeline(train_df, val_df, artifact_dir=artifact_dir)

    assert pipeline is not None
    assert "accuracy" in metrics
    assert "f1_score" in metrics
    assert os.path.exists(os.path.join(artifact_dir, "model.joblib"))
    assert os.path.exists(os.path.join(artifact_dir, "metrics.json"))
    assert os.path.exists(os.path.join(artifact_dir, "confusion_matrix.png"))
