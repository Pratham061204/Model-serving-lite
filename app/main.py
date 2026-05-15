# import os
# os.environ["USE_TF"] = "0"
# os.environ["USE_TORCH"] = "1"

# from fastapi import FastAPI
# from pydantic import BaseModel
# from transformers import pipeline
# from datetime import datetime
# from collections import Counter, deque
# import time
# import json
# import uuid
# import re

# app = FastAPI(title="Sentiment Service with Monitoring")

# if os.getenv("TEST_MODE") == "1":
#     def sentiment_model(text):
#         return [{"label": "POSITIVE", "score": 0.99}]
# else:
#     sentiment_model = pipeline(
#         "sentiment-analysis",
#         model="distilbert-base-uncased-finetuned-sst-2-english",
#         framework="pt"
#     )

# LOG_PATH = "logs/predictions.jsonl"
# os.makedirs("logs", exist_ok=True)

# # In-memory monitoring state for demo
# recent_lengths = deque(maxlen=50)
# recent_latencies = deque(maxlen=50)
# label_counter = Counter()

# BASELINE_VOCAB = {
#     "good", "bad", "great", "poor", "amazing", "terrible", "love", "hate",
#     "product", "service", "app", "experience", "happy", "sad", "slow",
#     "fast", "useful", "broken", "works", "issue"
# }


# class PredictRequest(BaseModel):
#     text: str


# def tokenize(text: str):
#     return set(re.findall(r"\b[a-zA-Z]+\b", text.lower()))


# def percentile(values, p):
#     if not values:
#         return 0
#     sorted_values = sorted(values)
#     index = int((p / 100) * (len(sorted_values) - 1))
#     return sorted_values[index]


# def detect_drift(text: str, label: str):
#     words = text.split()
#     word_count = len(words)
#     recent_lengths.append(word_count)
#     label_counter[label] += 1

#     tokens = tokenize(text)
#     vocab_overlap = len(tokens & BASELINE_VOCAB) / max(len(tokens), 1)

#     non_ascii_ratio = sum(1 for c in text if ord(c) > 127) / max(len(text), 1)
#     avg_recent_length = sum(recent_lengths) / len(recent_lengths)

#     drift_reasons = []

#     if word_count > 15:
#         drift_reasons.append("single_request_text_too_long")

#     if len(recent_lengths) >= 10 and avg_recent_length > 10:
#         drift_reasons.append("recent_average_text_length_high")

#     if vocab_overlap < 0.2 and len(tokens) >= 5:
#         drift_reasons.append("low_vocab_overlap_with_baseline")

#     if non_ascii_ratio > 0.2:
#         drift_reasons.append("possible_language_or_encoding_shift")

#     total_predictions = sum(label_counter.values())
#     if total_predictions >= 20:
#         dominant_label_count = max(label_counter.values())
#         if dominant_label_count / total_predictions > 0.9:
#             drift_reasons.append("label_distribution_imbalance")

#     return {
#         "drift_detected": len(drift_reasons) > 0,
#         "drift_reasons": drift_reasons,
#         "word_count": word_count,
#         "avg_recent_length": avg_recent_length,
#         "vocab_overlap": round(vocab_overlap, 3),
#         "non_ascii_ratio": round(non_ascii_ratio, 3)
#     }


# @app.get("/")
# def root():
#     return {
#         "message": "Sentiment API is running",
#         "docs": "/docs",
#         "health": "/health",
#         "metrics": "/metrics"
#     }


# @app.get("/health")
# def health():
#     return {"status": "ok"}


# @app.get("/metrics")
# def metrics():
#     latencies = list(recent_latencies)

#     return {
#         "total_predictions": sum(label_counter.values()),
#         "label_distribution": dict(label_counter),
#         "avg_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else 0,
#         "p95_latency_ms": round(percentile(latencies, 95), 2) if latencies else 0,
#         "avg_recent_text_length": round(sum(recent_lengths) / len(recent_lengths), 2) if recent_lengths else 0
#     }


# @app.post("/predict")
# def predict(request: PredictRequest):
#     request_id = str(uuid.uuid4())
#     start_time = time.perf_counter()

#     model_start = time.perf_counter()
#     result = sentiment_model(request.text)[0]
#     model_latency_ms = (time.perf_counter() - model_start) * 1000

#     label = result["label"]
#     score = float(result["score"])

#     drift_info = detect_drift(request.text, label)

#     total_latency_ms = (time.perf_counter() - start_time) * 1000
#     recent_latencies.append(total_latency_ms)

#     log_entry = {
#         "request_id": request_id,
#         "timestamp": datetime.utcnow().isoformat(),
#         "input_text": request.text,
#         "prediction": label,
#         "confidence": score,
#         "model_latency_ms": round(model_latency_ms, 2),
#         "total_latency_ms": round(total_latency_ms, 2),
#         "drift_detected": drift_info["drift_detected"],
#         "drift_reasons": drift_info["drift_reasons"],
#         "word_count": drift_info["word_count"],
#         "vocab_overlap": drift_info["vocab_overlap"],
#         "non_ascii_ratio": drift_info["non_ascii_ratio"]
#     }

#     with open(LOG_PATH, "a", encoding="utf-8") as f:
#         f.write(json.dumps(log_entry) + "\n")

#     return {
#         "request_id": request_id,
#         "label": label,
#         "score": score,
#         "model_latency_ms": round(model_latency_ms, 2),
#         "total_latency_ms": round(total_latency_ms, 2),
#         "drift": drift_info
#     }

import os
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"

from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
from datetime import datetime
from collections import Counter, deque
import time
import json
import uuid
import re

app = FastAPI(title="Sentiment Service with Monitoring")

_model = None

def get_model():
    global _model
    if os.getenv("TEST_MODE") == "1":
        return lambda text: [{"label": "POSITIVE", "score": 0.99}]
    if _model is None:
        _model = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            framework="pt"
        )
    return _model

LOG_PATH = "logs/predictions.jsonl"
os.makedirs("logs", exist_ok=True)

recent_lengths = deque(maxlen=50)
recent_latencies = deque(maxlen=50)
label_counter = Counter()

BASELINE_VOCAB = {
    "good", "bad", "great", "poor", "amazing", "terrible", "love", "hate",
    "product", "service", "app", "experience", "happy", "sad", "slow",
    "fast", "useful", "broken", "works", "issue"
}


class PredictRequest(BaseModel):
    text: str


def tokenize(text: str):
    return set(re.findall(r"\b[a-zA-Z]+\b", text.lower()))


def percentile(values, p):
    if not values:
        return 0
    sorted_values = sorted(values)
    index = int((p / 100) * (len(sorted_values) - 1))
    return sorted_values[index]


def detect_drift(text: str, label: str):
    words = text.split()
    word_count = len(words)
    recent_lengths.append(word_count)
    label_counter[label] += 1

    tokens = tokenize(text)
    vocab_overlap = len(tokens & BASELINE_VOCAB) / max(len(tokens), 1)

    non_ascii_ratio = sum(1 for c in text if ord(c) > 127) / max(len(text), 1)
    avg_recent_length = sum(recent_lengths) / len(recent_lengths)

    drift_reasons = []

    if word_count > 15:
        drift_reasons.append("single_request_text_too_long")

    if len(recent_lengths) >= 10 and avg_recent_length > 10:
        drift_reasons.append("recent_average_text_length_high")

    if vocab_overlap < 0.2 and len(tokens) >= 5:
        drift_reasons.append("low_vocab_overlap_with_baseline")

    if non_ascii_ratio > 0.2:
        drift_reasons.append("possible_language_or_encoding_shift")

    total_predictions = sum(label_counter.values())
    if total_predictions >= 20:
        dominant_label_count = max(label_counter.values())
        if dominant_label_count / total_predictions > 0.9:
            drift_reasons.append("label_distribution_imbalance")

    return {
        "drift_detected": len(drift_reasons) > 0,
        "drift_reasons": drift_reasons,
        "word_count": word_count,
        "avg_recent_length": avg_recent_length,
        "vocab_overlap": round(vocab_overlap, 3),
        "non_ascii_ratio": round(non_ascii_ratio, 3)
    }


@app.get("/")
def root():
    return {
        "message": "Sentiment API is running",
        "docs": "/docs",
        "health": "/health",
        "metrics": "/metrics"
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/metrics")
def metrics():
    latencies = list(recent_latencies)
    return {
        "total_predictions": sum(label_counter.values()),
        "label_distribution": dict(label_counter),
        "avg_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else 0,
        "p95_latency_ms": round(percentile(latencies, 95), 2) if latencies else 0,
        "avg_recent_text_length": round(sum(recent_lengths) / len(recent_lengths), 2) if recent_lengths else 0
    }


@app.post("/predict")
def predict(request: PredictRequest):
    request_id = str(uuid.uuid4())
    start_time = time.perf_counter()

    model_start = time.perf_counter()
    result = get_model()(request.text)[0]
    model_latency_ms = (time.perf_counter() - model_start) * 1000

    label = result["label"]
    score = float(result["score"])

    drift_info = detect_drift(request.text, label)

    total_latency_ms = (time.perf_counter() - start_time) * 1000
    recent_latencies.append(total_latency_ms)

    log_entry = {
        "request_id": request_id,
        "timestamp": datetime.utcnow().isoformat(),
        "input_text": request.text,
        "prediction": label,
        "confidence": score,
        "model_latency_ms": round(model_latency_ms, 2),
        "total_latency_ms": round(total_latency_ms, 2),
        "drift_detected": drift_info["drift_detected"],
        "drift_reasons": drift_info["drift_reasons"],
        "word_count": drift_info["word_count"],
        "vocab_overlap": drift_info["vocab_overlap"],
        "non_ascii_ratio": drift_info["non_ascii_ratio"]
    }

    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry) + "\n")

    return {
        "request_id": request_id,
        "label": label,
        "score": score,
        "model_latency_ms": round(model_latency_ms, 2),
        "total_latency_ms": round(total_latency_ms, 2),
        "drift": drift_info
    }