"""
Training Pipeline for Disaster Tweets Classification.

Loads dataset, cleans text, performs train/validation split, trains classification
model (TF-IDF + Scikit-Learn baseline or Transformer fine-tuning), evaluates performance
(Accuracy, F1, Precision, Recall, Confusion Matrix), and persists all artifacts.
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline

# Ensure src modules are importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.prepare_data import prepare_dataset
from src.preprocessing import PreprocessingConfig, TextCleaner, build_tfidf_vectorizer


def evaluate_model(
    model: Any,
    X_val: Any,
    y_val: np.ndarray,
    artifact_dir: str = "artifacts"
) -> Dict[str, Any]:
    """
    Evaluates model performance on validation data and exports confusion matrix plot.

    Args:
        model: Trained model or pipeline object.
        X_val: Feature matrix or text series for evaluation.
        y_val (np.ndarray): True target ground truth labels.
        artifact_dir (str): Output directory for evaluation reports.

    Returns:
        Dict[str, Any]: Calculated evaluation metrics dictionary.
    """
    y_pred = model.predict(X_val)

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_val)[:, 1]
        try:
            auc = float(roc_auc_score(y_val, y_proba))
        except Exception:
            auc = 0.0
    else:
        y_proba = None
        auc = 0.0

    acc = float(accuracy_score(y_val, y_pred))
    precision = float(precision_score(y_val, y_pred, average="binary", zero_division=0))
    recall = float(recall_score(y_val, y_pred, average="binary", zero_division=0))
    f1 = float(f1_score(y_val, y_pred, average="binary", zero_division=0))

    cm = confusion_matrix(y_val, y_pred).tolist()

    metrics = {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": auc,
        "confusion_matrix": cm,
        "classification_report": classification_report(y_val, y_pred, output_dict=True)
    }

    os.makedirs(artifact_dir, exist_ok=True)

    # Plot & Save Confusion Matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Not Disaster", "Disaster"],
        yticklabels=["Not Disaster", "Disaster"]
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    cm_path = os.path.join(artifact_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()

    metrics_path = os.path.join(artifact_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n================ MODEL EVALUATION ================")
    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-Score  : {f1:.4f}")
    print(f"ROC AUC   : {auc:.4f}")
    print(f"Saved evaluation metrics to '{metrics_path}'")
    print(f"Saved confusion matrix plot to '{cm_path}'")
    print(f"==================================================\n")

    return metrics


def train_baseline_pipeline(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    artifact_dir: str = "artifacts"
) -> Tuple[Any, Dict[str, Any]]:
    """
    Trains an optimized TF-IDF + Logistic Regression / Calibrated Classifier baseline model.

    Args:
        train_df (pd.DataFrame): Training dataframe with 'text' and 'target'.
        val_df (pd.DataFrame): Validation dataframe with 'text' and 'target'.
        artifact_dir (str): Artifact persistence directory.

    Returns:
        Tuple[Any, Dict[str, Any]]: Trained pipeline and evaluation metrics.
    """
    print("[Train] Preprocessing text data...")
    cleaner = TextCleaner()

    X_train_raw = train_df["text"].fillna("").astype(str).tolist()
    y_train = train_df["target"].values

    X_val_raw = val_df["text"].fillna("").astype(str).tolist()
    y_val = val_df["target"].values

    X_train_clean = cleaner.clean_batch(X_train_raw)
    X_val_clean = cleaner.clean_batch(X_val_raw)

    print("[Train] Building vectorizer & classification pipeline...")
    vectorizer = build_tfidf_vectorizer(max_features=12000, ngram_range=(1, 2))

    classifier = LogisticRegression(
        C=2.0,
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    )

    pipeline = Pipeline([
        ("tfidf", vectorizer),
        ("clf", classifier)
    ])

    print("[Train] Fitting model on clean training tweets...")
    pipeline.fit(X_train_clean, y_train)

    print("[Train] Evaluating model on validation split...")
    metrics = evaluate_model(pipeline, X_val_clean, y_val, artifact_dir=artifact_dir)

    # Save artifacts
    os.makedirs(artifact_dir, exist_ok=True)

    model_path = os.path.join(artifact_dir, "model.joblib")
    joblib.dump(pipeline, model_path)

    target_map = {"0": "Not Disaster", "1": "Disaster"}
    target_map_path = os.path.join(artifact_dir, "target_map.json")
    with open(target_map_path, "w") as f:
        json.dump(target_map, f, indent=2)

    metadata = {
        "model_type": "tfidf_logistic_regression",
        "n_train_samples": len(train_df),
        "n_val_samples": len(val_df),
        "vocabulary_size": len(pipeline.named_steps["tfidf"].vocabulary_),
        "metrics": {
            "accuracy": metrics["accuracy"],
            "f1_score": metrics["f1_score"],
            "roc_auc": metrics["roc_auc"]
        }
    }
    metadata_path = os.path.join(artifact_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"[Train] Saved model pipeline to '{model_path}'")
    return pipeline, metrics


def run_training(
    data_dir: str = "data",
    artifact_dir: str = "artifacts",
    test_size: float = 0.2,
    random_state: int = 42
) -> None:
    """
    Main training workflow manager.
    """
    train_path, test_path = prepare_dataset(data_dir=data_dir, test_size=test_size, random_state=random_state)

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    print(f"[Train] Loaded dataset with {len(train_df)} train samples and {len(test_df)} test samples.")

    train_baseline_pipeline(train_df, test_df, artifact_dir=artifact_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Disaster Tweets Classifier")
    parser.add_argument("--data-dir", type=str, default="data", help="Directory containing dataset files")
    parser.add_argument("--artifact-dir", type=str, default="artifacts", help="Directory to save model artifacts")
    parser.add_argument("--test-size", type=float, default=0.2, help="Validation set split ratio")
    args = parser.parse_args()

    run_training(data_dir=args.data_dir, artifact_dir=args.artifact_dir, test_size=args.test_size)
