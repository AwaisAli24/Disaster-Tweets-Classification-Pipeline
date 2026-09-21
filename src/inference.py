"""
Inference Module for Disaster Tweets Classification.

Provides low-latency model inference by loading trained pipeline artifacts once at initialization.
Supports raw string input cleaning, confidence scoring, probability outputs, and batch predictions.
"""

import json
import os
import sys
from typing import Any, Dict, List, Union

import joblib
import numpy as np

# Ensure src modules are importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocessing import PreprocessingConfig, TextCleaner


class DisasterTweetPredictor:
    """
    Production-ready predictor class. Loads trained model artifacts once on startup
    and exposes fast single & batch inference methods.
    """

    def __init__(self, artifact_dir: str = "artifacts"):
        self.artifact_dir = artifact_dir
        self.model_path = os.path.join(artifact_dir, "model.joblib")
        self.target_map_path = os.path.join(artifact_dir, "target_map.json")

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model artifact not found at '{self.model_path}'. Please run training script first."
            )

        # Load artifacts ONCE at startup
        print(f"[Inference Engine] Loading model weights from '{self.model_path}'...")
        self.model = joblib.load(self.model_path)

        if os.path.exists(self.target_map_path):
            with open(self.target_map_path, "r") as f:
                self.target_map = json.load(f)
        else:
            self.target_map = {"0": "Not Disaster", "1": "Disaster"}

        self.cleaner = TextCleaner()
        print("[Inference Engine] Initialization complete. Model loaded into memory.")

    def predict(self, text: str) -> Dict[str, Any]:
        """
        Classifies a single input tweet string.

        Args:
            text (str): Raw tweet text string.

        Returns:
            Dict[str, Any]: Structured prediction output dictionary.
        """
        if not isinstance(text, str) or not text.strip():
            return {
                "text": text,
                "cleaned_text": "",
                "label": "Not Disaster",
                "label_id": 0,
                "confidence": 0.5,
                "probabilities": {"Not Disaster": 0.5, "Disaster": 0.5}
            }

        cleaned_text = self.cleaner.clean_text(text)

        # Predict label & probability array
        pred_label_id = int(self.model.predict([cleaned_text])[0])

        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba([cleaned_text])[0]
            prob_not_disaster = float(probs[0])
            prob_disaster = float(probs[1])
        else:
            prob_disaster = 1.0 if pred_label_id == 1 else 0.0
            prob_not_disaster = 1.0 - prob_disaster

        label_name = self.target_map.get(str(pred_label_id), "Disaster" if pred_label_id == 1 else "Not Disaster")
        confidence = prob_disaster if pred_label_id == 1 else prob_not_disaster

        return {
            "text": text,
            "cleaned_text": cleaned_text,
            "label": label_name,
            "label_id": pred_label_id,
            "confidence": round(confidence, 4),
            "probabilities": {
                "Not Disaster": round(prob_not_disaster, 4),
                "Disaster": round(prob_disaster, 4)
            }
        }

    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Classifies a list of raw tweet strings.

        Args:
            texts (List[str]): List of raw tweet strings.

        Returns:
            List[Dict[str, Any]]: List of prediction result dictionaries.
        """
        if not texts:
            return []

        cleaned_texts = [self.cleaner.clean_text(t) for t in texts]
        pred_label_ids = self.model.predict(cleaned_texts)

        if hasattr(self.model, "predict_proba"):
            probs_matrix = self.model.predict_proba(cleaned_texts)
        else:
            probs_matrix = np.zeros((len(texts), 2))
            for i, p_id in enumerate(pred_label_ids):
                probs_matrix[i, p_id] = 1.0

        results = []
        for i in range(len(texts)):
            pred_label_id = int(pred_label_ids[i])
            p_not_disaster = float(probs_matrix[i, 0])
            p_disaster = float(probs_matrix[i, 1])

            label_name = self.target_map.get(str(pred_label_id), "Disaster" if pred_label_id == 1 else "Not Disaster")
            confidence = p_disaster if pred_label_id == 1 else p_not_disaster

            results.append({
                "text": texts[i],
                "cleaned_text": cleaned_texts[i],
                "label": label_name,
                "label_id": pred_label_id,
                "confidence": round(confidence, 4),
                "probabilities": {
                    "Not Disaster": round(p_not_disaster, 4),
                    "Disaster": round(p_disaster, 4)
                }
            })

        return results
