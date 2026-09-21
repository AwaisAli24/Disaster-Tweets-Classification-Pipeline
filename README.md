# Disaster Tweets Classification Pipeline

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E.svg)](https://scikit-learn.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)

A complete, modular, and production-ready text classification pipeline that predicts whether a social media tweet describes a real-world disaster or an everyday event using Natural Language Processing (NLP), Scikit-Learn, PyTorch / Hugging Face Transformers, and FastAPI.

---

## 📌 Features

- **Robust Social Media Preprocessing (`src/preprocessing.py`)**: Dedicated `TextCleaner` for stripping Twitter handles (`@user`), URLs, HTML entities (`&amp;`), hashtag symbols (`#`), non-printable characters, and normalizing whitespace.
- **Modular Model Training (`src/train.py`)**: Supports data splitting, training TF-IDF + Logistic Regression / Calibrated baseline models, evaluation metric suite (Accuracy, Precision, Recall, F1-Score, ROC-AUC), and visual Confusion Matrix export (`artifacts/confusion_matrix.png`).
- **Production Inference Engine (`src/inference.py`)**: Low-latency `DisasterTweetPredictor` that loads model weights **once** during initialization to support single prompt and batch predictions without per-request reloading overhead.
- **RESTful FastAPI Service (`api/main.py`)**: Exposes `/health`, `/predict`, and `/predict/batch` endpoints using FastAPI lifespan event handlers, Pydantic data validation, and latency tracking.
- **Comprehensive Unit & Integration Test Suite (`tests/`)**: Automated `pytest` suite validating preprocessing, training, inference, and API endpoints.

---

## 📂 Project Directory Structure

```
Disaster Tweets Classification Pipeline/
├── data/
│   ├── DisasterTweets.csv       # Raw input dataset
│   ├── prepare_data.py          # Data standardization & splitting script
│   └── README.md                # Data documentation
├── src/
│   ├── preprocessing.py         # Text cleaning utilities & TF-IDF builder
│   ├── train.py                 # Training workflow & evaluation metrics
│   └── inference.py             # Production predictor class
├── api/
│   ├── main.py                  # FastAPI REST API application
│   └── schemas.py               # Pydantic request/response schemas
├── tests/
│   ├── test_preprocessing.py    # Unit tests for text cleaning
│   ├── test_train.py            # Integration tests for model training
│   ├── test_inference.py        # Tests for inference predictor
│   └── test_api.py              # API endpoint integration tests
├── artifacts/                   # Saved models, metrics.json & confusion_matrix.png
├── requirements.txt             # Pinned Python dependencies
└── README.md                    # Project guide and instructions
```

---

## ⚡ Quick Start & Installation

### 1. Environment Setup

Create and activate a virtual environment:

```bash
# Create Python 3.12 virtual environment
python3 -m venv .venv

# Activate virtual environment (macOS/Linux)
source .venv/bin/activate

# Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🚀 Running the Pipeline

### Step 1: Prepare and Standardize Dataset

Format raw dataset into standardized `train.csv` and `test.csv` files:

```bash
python data/prepare_data.py
```

### Step 2: Train Model & Evaluate

Train the text classification model, compute metrics, and export artifacts (`model.joblib`, `metrics.json`, `confusion_matrix.png`):

```bash
python src/train.py
```

**Expected Output:**
```
================ MODEL EVALUATION ================
Accuracy  : 0.9414
Precision : 0.9520
Recall    : 0.9300
F1-Score  : 0.9409
ROC AUC   : 0.9785
Saved evaluation metrics to 'artifacts/metrics.json'
Saved confusion matrix plot to 'artifacts/confusion_matrix.png'
==================================================
```

---

## 🌐 Launching the REST API Server

Start the production FastAPI server using Uvicorn:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Interactive API documentation will be available at:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

---

## 📡 API Usage & Example Requests

### 1. Single Tweet Prediction (`POST /predict`)

**cURL Request:**
```bash
curl -X 'POST' \
  'http://localhost:8000/predict' \
  -H 'Content-Type: application/json' \
  -d '{
  "text": "Emergency evacuation ordered in downtown main street due to wildfire! #wildfire"
}'
```

**JSON Response:**
```json
{
  "text": "Emergency evacuation ordered in downtown main street due to wildfire! #wildfire",
  "cleaned_text": "emergency evacuation ordered in downtown main street due to wildfire! wildfire",
  "label": "Disaster",
  "label_id": 1,
  "confidence": 0.9432,
  "probabilities": {
    "Not Disaster": 0.0568,
    "Disaster": 0.9432
  },
  "latency_ms": 1.45
}
```

### 2. Batch Tweet Predictions (`POST /predict/batch`)

**cURL Request:**
```bash
curl -X 'POST' \
  'http://localhost:8000/predict/batch' \
  -H 'Content-Type: application/json' \
  -d '{
  "tweets": [
    "Severe flood warning issued for coastal areas #flood",
    "Enjoying a delicious coffee break at the central park cafe"
  ]
}'
```

**JSON Response:**
```json
{
  "predictions": [
    {
      "text": "Severe flood warning issued for coastal areas #flood",
      "cleaned_text": "severe flood warning issued for coastal areas flood",
      "label": "Disaster",
      "label_id": 1,
      "confidence": 0.9612,
      "probabilities": {
        "Not Disaster": 0.0388,
        "Disaster": 0.9612
      }
    },
    {
      "text": "Enjoying a delicious coffee break at the central park cafe",
      "cleaned_text": "enjoying a delicious coffee break at the central park cafe",
      "label": "Not Disaster",
      "label_id": 0,
      "confidence": 0.9785,
      "probabilities": {
        "Not Disaster": 0.9785,
        "Disaster": 0.0215
      }
    }
  ],
  "total_tweets": 2,
  "latency_ms": 2.31
}
```

---

## 🧪 Running Unit & Integration Tests

Run the complete automated test suite using `pytest`:

```bash
pytest tests/ -v
```

---

## 📊 Evaluation & Artifacts

All training execution artifacts are stored in `artifacts/`:
- `model.joblib`: Serialized vectorizer + classification model pipeline.
- `target_map.json`: Label ID mapping (`{"0": "Not Disaster", "1": "Disaster"}`).
- `metrics.json`: JSON output containing accuracy, F1, precision, recall, ROC-AUC, and classification report.
- `confusion_matrix.png`: High-resolution visual plot of validation set confusion matrix.
