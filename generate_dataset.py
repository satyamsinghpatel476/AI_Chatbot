import json
import random

robotics = [
    "Explain SLAM", "Explain Kalman Filter", "Explain PID controller",
    "What is robot localization?", "Explain sensor fusion"
]

daily = [
    "Suggest food apps", "Recommend travel apps",
    "Give workout plan", "Suggest study tips"
]

personal = [
    "My name is Rahul", "Remember I like coffee",
    "I prefer tea", "My hobby is coding"
]

dataset = []

# Single intents
for r in robotics:
    dataset.append({"text": r, "label": "robotics"})

for d in daily:
    dataset.append({"text": d, "label": "daily"})

for p in personal:
    dataset.append({"text": p, "label": "personal"})

# Mixed generation
for _ in range(300):
    q = random.choice(robotics) + " and " + random.choice(daily)
    dataset.append({"text": q, "label": "mixed"})

    q = random.choice(personal) + " and " + random.choice(robotics)
    dataset.append({"text": q, "label": "mixed"})

# Save
with open("data/intent_dataset.json", "w") as f:
    json.dump(dataset, f, indent=2)

print("🔥 Dataset generated")
