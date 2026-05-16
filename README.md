---
title: Case9 Model Serving Lite
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# Case 9: Model Serving Lite

**Live demo:** https://pratham06-case9-model-serving-lite.hf.space/docs 
**API docs:** https://pratham06-case9-model-serving-lite.hf.space/docs#/default/predict_predict_post 
**Repo:** https://github.com/Pratham061204/Model-serving-lite 
**Demo video:** <https://www.loom.com/share/57faf1db51304d97b3ed2fe934615aff>

## Demo GIF

![Case 9 demo](assets/case9-demo.gif)

## What this is

This project converts a notebook-style sentiment classifier into a deployable, monitored API service.

It exposes a `/predict` endpoint, logs each prediction, tracks latency, detects simple input drift, and includes CI checks plus a retrain/evaluation gate.

## Core endpoints

| Endpoint | Purpose |
|---|---|
| `/` | Basic service information |
| `/health` | Health check for deployment |
| `/predict` | Sentiment prediction endpoint |
| `/metrics` | Runtime metrics such as prediction count, label distribution, average latency, p95 latency, and recent text length |

## How to test the `/predict` API

The `/predict` endpoint is a **POST** endpoint, so it will not return a prediction by simply opening `/predict` in a browser.

### Option 1: Test from Swagger UI

Open the direct API docs link:

```text
https://pratham06-case9-model-serving-lite.hf.space/docs#/default/predict_predict_post
```

Then:

1. Click **Try it out**
2. Paste any text in the request body, for example:

```json
{
  "text": "I love this app, it is amazing"
}
```

3. Click **Execute**
4. The API will return the sentiment label, confidence score, latency, request ID, and drift information.

### Option 2: Test with Postman or Insomnia

Use:

```text
POST https://pratham06-case9-model-serving-lite.hf.space/predict
```

Headers:

```text
Content-Type: application/json
```

Body:

```json
{
  "text": "I love this app, it is amazing"
}
```

### Option 3: Test with PowerShell

```powershell
Invoke-RestMethod `
  -Uri "https://pratham06-case9-model-serving-lite.hf.space/predict" `
  -Method POST `
  -ContentType "application/json" `
  -Body '{"text":"I love this app, it is amazing"}'
```

### Option 4: Test with curl

```bash
curl -X POST "https://pratham06-case9-model-serving-lite.hf.space/predict" \
  -H "Content-Type: application/json" \
  -d '{"text":"I love this app, it is amazing"}'
```

## Example response

```json
{
  "request_id": "generated-request-id",
  "label": "POSITIVE",
  "score": 0.99,
  "model_latency_ms": 123.45,
  "total_latency_ms": 124.12,
  "drift": {
    "drift_detected": false,
    "drift_reasons": [],
    "word_count": 8,
    "avg_recent_length": 8.0,
    "vocab_overlap": 0.5,
    "non_ascii_ratio": 0.0
  }
}
```

## How to run locally

```powershell
git clone <PASTE_YOUR_GITHUB_REPO_LINK>
cd case9-model-serving-lite

python -m venv .venv
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
uvicorn app.main:app --reload
```

Then open:

```text
http://127.0.0.1:8000/docs
```

## How to run with Docker

```powershell
docker build -t sentiment-service .
docker run -p 7860:7860 sentiment-service
```

Then open:

```text
http://127.0.0.1:7860/docs
```

## How to run tests

```powershell
$env:TEST_MODE="1"
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD="1"
python -m pytest -q
```

Expected result:

```text
3 passed
```

## Stack

| Tool | Why used |
|---|---|
| FastAPI | Lightweight API framework with automatic Swagger docs |
| Hugging Face Transformers | Pretrained sentiment model for reliable inference |
| PyTorch CPU | Runs without GPU and avoids CUDA dependency issues |
| Docker | Makes the service reproducible across local and cloud environments |
| GitHub Actions | Runs tests and Docker build checks on every push |
| JSONL logging | Simple request/response audit trail for debugging predictions |
| Hugging Face Spaces | Free CPU deployment suitable for ML demos |

## Monitoring and failure detection

The service logs each prediction with:

- request ID
- timestamp
- input text
- predicted label
- confidence score
- model latency
- total latency
- drift status
- drift reasons
- text length
- vocabulary overlap
- non-ASCII ratio

The `/metrics` endpoint exposes basic runtime signals such as:

- total prediction count
- label distribution
- average latency
- p95 latency
- recent average input length

## Drift detection approach

The prototype checks for:

- unusually long input text
- rising average text length
- low vocabulary overlap with expected sentiment vocabulary
- possible language or encoding shift using non-ASCII ratio
- label distribution imbalance

This is intentionally lightweight, but it demonstrates how a production service could flag input/model mismatch early.

## Retrain and evaluation workflow

The retrain workflow is implemented as a lightweight CI promotion gate.

When training or evaluation files change, the workflow:

1. runs a retrain step,
2. evaluates against a held-out dataset,
3. compares candidate performance against a baseline threshold,
4. rejects the candidate if metrics regress.

For this one-day prototype, full transformer fine-tuning was de-scoped because it is too heavy for free CPU CI. In production, this workflow would be replaced with full fine-tuning, versioned model artifacts, and a model registry.

## CI/CD

The repo includes two GitHub Actions workflows:

| Workflow | Purpose |
|---|---|
| `ci.yml` | Runs API tests and builds the Docker image |
| `retrain.yml` | Runs retrain/evaluation gate and rejects regressed candidates |

## What's NOT done

- Full transformer fine-tuning is not implemented because this is a one-day free-CPU prototype.
- Logs are written locally as JSONL; production should use persistent log storage.
- Drift detection is heuristic-based, not full statistical monitoring.
- No authentication is added because the demo is focused on model serving and MLOps workflow.
- No model registry is included, but the retrain gate is structured so one can be added later.

## In production, I would also add

- persistent logging using CloudWatch, Datadog, Grafana Loki, or ELK
- Prometheus metrics and alerting
- model registry with versioned artifacts
- canary or shadow deployment for candidate models
- authentication and rate limiting
- persistent storage for prediction logs
- automated rollback if latency, error rate, or model quality crosses thresholds

## Known deployment note

The Docker image is larger than a normal web API because it includes PyTorch and Transformers. I used CPU-only PyTorch to avoid unnecessary CUDA/NVIDIA dependencies and make deployment more suitable for free-tier CPU hosts.

The Hugging Face Space uses port `7860`, which is why the Dockerfile exposes `7860` and the README front matter uses `app_port: 7860`.
