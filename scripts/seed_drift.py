import requests

base_url = "http://localhost:8000"

normal = [
    "Good product", "Loved it", "Terrible quality", "Would not buy again",
    "Fast shipping", "Broke after a week", "Amazing value", "Disappointing"
]

drifted = [
    "The molecular composition of this material degrades significantly under standard atmospheric pressure conditions resulting in catastrophic structural failure",
    "Upon extensive longitudinal analysis of the product lifecycle management protocols the systemic inadequacies become increasingly apparent",
    "Comprehensive evaluation of the thermodynamic properties suggests fundamental engineering oversights in the manufacturing specification documentation",
]

print("Seeding baseline...")
for text in normal * 5:
    r = requests.post(f"{base_url}/predict", json={"text": text})
    print(f"  drift_detected: {r.json()['drift']['drift_detected']}")

print("\nSeeding drift...")
for text in drifted * 7:
    r = requests.post(f"{base_url}/predict", json={"text": text})
    d = r.json()['drift']
    print(f"  drift_detected: {d['drift_detected']} | reasons: {d['drift_reasons']}")

print("\n--- Final metrics ---")
import json
m = requests.get(f"{base_url}/metrics").json()
print(json.dumps(m, indent=2))