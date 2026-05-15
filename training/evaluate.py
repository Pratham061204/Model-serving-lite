import csv
import json
import os
import sys
from transformers import pipeline

BASELINE_ACCURACY = 0.80
REPORT_PATH = "training/evaluation_report.json"

def load_data(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows

def main():
    holdout_path = "data/holdout.csv"

    if not os.path.exists(holdout_path):
        print("Missing holdout dataset.")
        sys.exit(1)

    classifier = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        framework="pt"
    )

    rows = load_data(holdout_path)

    correct = 0
    predictions = []

    for row in rows:
        result = classifier(row["text"])[0]
        predicted_label = result["label"]
        expected_label = row["label"]

        is_correct = predicted_label == expected_label
        correct += int(is_correct)

        predictions.append({
            "text": row["text"],
            "expected": expected_label,
            "predicted": predicted_label,
            "confidence": float(result["score"]),
            "correct": is_correct
        })

    accuracy = correct / len(rows)

    report = {
        "baseline_accuracy": BASELINE_ACCURACY,
        "candidate_accuracy": accuracy,
        "promote": accuracy >= BASELINE_ACCURACY,
        "predictions": predictions
    }

    os.makedirs("training", exist_ok=True)

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))

    if accuracy < BASELINE_ACCURACY:
        print("Regression detected. Candidate model rejected.")
        sys.exit(1)

    print("Candidate model accepted. Metrics did not regress.")

if __name__ == "__main__":
    main()