# Decisions Log — Case 9: Model Serving Lite

## Assumptions I made

1. The main goal is to demonstrate production-readiness around model serving, not to maximize sentiment-model accuracy.
2. A pretrained Hugging Face sentiment model is acceptable because the case allows any pretrained or open model.
3. Free CPU infrastructure is enough for the prototype, so I avoided GPU-specific dependencies.
4. Lightweight heuristic drift detection is acceptable for a one-day prototype, as long as the limitations are documented.
5. The retrain workflow should prioritize demonstrating the promotion-gate pattern over expensive fine-tuning.
6. Local JSONL logging is acceptable for a prototype, but production would need centralized logging.
7. Authentication was not required for the demo because the focus is MLOps workflow, deployment, and monitoring.

## Trade-offs

| Choice | Alternative | Why I picked this |
|---|---|---|
| FastAPI | Flask | FastAPI gives automatic `/docs`, request validation, and a cleaner API structure. |
| Hugging Face pipeline | Custom trained model | Faster to ship and directly aligned with the brief's pretrained-model suggestion. |
| DistilBERT SST-2 model | Larger transformer model | Small enough for CPU demo, still reliable for sentiment classification. |
| CPU-only PyTorch | CUDA PyTorch | Avoids large NVIDIA dependencies and makes deployment simpler on free-tier hosts. |
| Docker | Local Python-only deployment | Docker makes the service reproducible and easier to deploy consistently. |
| JSONL logs | Database logging | Simple, inspectable, and enough for a prototype/debug demo. |
| Heuristic drift stub | Evidently/WhyLabs/statistical drift tooling | Lightweight and easy to explain within a one-day case. |
| Hugging Face Spaces | Render | Hugging Face is better suited for ML demos and free CPU model-serving prototypes. |
| Port 7860 on Hugging Face | Port 8000 | Hugging Face Docker Spaces expect the exposed app port to match the Space configuration, so I aligned the Dockerfile and README with `7860`. |
| Lightweight retrain gate | Full transformer fine-tuning | Full fine-tuning is too heavy for free CI and a one-day build. The promotion gate pattern is still demonstrated. |
| No authentication | API key/JWT auth | Authentication would be important in production, but it was de-scoped to focus on MLOps requirements. |

## What I de-scoped and why

- **Full transformer fine-tuning** — too slow and resource-heavy for the time-box and free CPU CI.
- **Persistent production logging** — local JSONL is enough for the demo; production should use centralized logging.
- **Authentication** — useful in production, but not necessary for showing model serving and monitoring.
- **Statistical drift monitoring** — heuristic checks are sufficient to demonstrate the concept.
- **Model registry** — documented as a production improvement.
- **Canary deployment** — useful, but outside the minimum one-day prototype scope.
- **Database storage for predictions** — local logs are simpler and easier to inspect during demo.

## Retrain workflow decision

The case asks for a retrain-and-promote workflow on new training-data PRs.

Because this is a one-day prototype running on free CPU CI, I implemented a lightweight retrain/evaluation gate rather than full transformer fine-tuning.

The workflow:

1. detects changes to `data/**` or `training/**`,
2. runs a retrain step,
3. evaluates the candidate against a held-out set,
4. compares candidate performance against a baseline threshold,
5. rejects the candidate if metrics regress.

In production, I would replace the lightweight retrain step with actual fine-tuning or scheduled batch retraining, store versioned model artifacts, and use a model registry.

## How I would know the model is failing before customers do

I would monitor the following signals:

### 1. Latency spikes

If p95 latency increases, the service may be facing cold starts, model loading issues, memory pressure, or infrastructure problems.

In this prototype:

- `model_latency_ms` is logged per request.
- `total_latency_ms` is logged per request.
- `/metrics` exposes average latency and p95 latency.

### 2. Confidence drops

If the model's confidence starts dropping over time, the model may be seeing unfamiliar inputs.

In production, I would track confidence distributions and alert if average confidence drops below a threshold.

### 3. Input drift

The model may fail if production text starts looking different from expected sentiment-review-style text.

This prototype checks:

- text length
- recent average text length
- vocabulary overlap
- non-ASCII ratio
- possible language or encoding shift

### 4. Label distribution imbalance

If predictions suddenly become mostly positive or mostly negative, the input distribution or model behavior may have shifted.

This prototype tracks label distribution in memory and exposes it through `/metrics`.

### 5. Error rate

Failed `/predict` requests should trigger alerts before users report issues.

In production, I would track HTTP 4xx/5xx rates and add alerts for sudden spikes.

## Monitoring signals included in this prototype

| Signal | Where it appears |
|---|---|
| Request ID | JSONL logs and prediction response |
| Prediction label | JSONL logs and response |
| Confidence score | JSONL logs and response |
| Model latency | JSONL logs and response |
| Total latency | JSONL logs and response |
| Drift status | JSONL logs and response |
| Drift reasons | JSONL logs and response |
| Label distribution | `/metrics` |
| p95 latency | `/metrics` |
| Average recent text length | `/metrics` |

## What I would do differently with another day

- Add Prometheus metrics and Grafana dashboard.
- Add persistent log storage.
- Add a real model registry with versioning.
- Add shadow deployment for candidate models.
- Add a small dashboard to visualize drift and latency trends.
- Add API authentication and rate limiting.
- Add alert thresholds for p95 latency, error rate, and drift rate.
- Store prediction logs in a database or object storage.
- Add automated rollback if a candidate model performs worse than production.

## AI assistance disclosure

I used AI assistants for debugging, implementation planning, and documentation drafting. I reviewed the generated code, tested it locally, and made the final design decisions myself.