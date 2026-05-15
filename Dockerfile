FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    torch==2.3.0

RUN pip install --no-cache-dir \
    fastapi==0.111.0 \
    uvicorn[standard]==0.29.0 \
    transformers==4.40.0 \
    pydantic==2.7.0

RUN python -c "from transformers import pipeline; \
    pipeline('sentiment-analysis', \
    model='distilbert-base-uncased-finetuned-sst-2-english', \
    framework='pt')"

COPY . .

RUN mkdir -p logs

EXPOSE 7860

CMD python -m uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860}