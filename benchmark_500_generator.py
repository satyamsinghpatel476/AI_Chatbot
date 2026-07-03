import json, csv
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

questions = []

def add(category, text):
    questions.append({
        "id": len(questions) + 1,
        "category": category,
        "question": text
    })

robotics_topics = [
    "SLAM", "LiDAR mapping", "AMCL localization", "EKF sensor fusion",
    "wheel odometry", "visual SLAM", "costmap inflation", "path planning",
    "loop closure", "robot navigation", "IMU drift", "ROS2 navigation",
    "dynamic obstacle avoidance", "robot localization", "sensor calibration",
    "map resolution", "wheel encoder faults", "robot footprint",
    "global planner", "local planner", "TF frames", "occupancy grid",
    "camera-based navigation", "GPS-denied navigation", "multi-sensor fusion"
]

robotics_templates = [
    "Why does {} fail in real-world environments?",
    "How can a beginner debug {}?",
    "What are the common causes of error in {}?",
    "How can {} improve autonomous robot performance?",
    "What are the limitations of {}?"
]

daily_topics = [
    "ride-sharing apps", "food delivery apps", "mobile GPS apps",
    "online privacy", "screen time reduction", "digital notes",
    "cloud storage", "public Wi-Fi", "mobile battery usage",
    "online course selection", "fake reviews", "app permissions",
    "QR payments", "online shopping", "password managers",
    "study planning", "phone storage", "browser extensions",
    "file backup", "navigation apps", "mobile data usage",
    "app updates", "student productivity", "online scams",
    "safe downloads"
]

daily_templates = [
    "How can a beginner choose between different {}?",
    "What risks should I consider while using {}?",
    "How can I improve safety and privacy while using {}?",
    "Why do people get different results when using {}?",
    "What practical steps can improve my experience with {}?"
]

ambiguous_questions = [
    "How can I improve tracking accuracy?",
    "Why is my system drifting over time?",
    "How do I reduce latency?",
    "What is the best way to reach my destination?",
    "How can I improve mapping reliability?",
    "Why does my navigation fail after an update?",
    "How can I detect fake signals?",
    "What should I do when localization is wrong?",
    "How do I choose the best sensor?",
    "Why does the assistant give wrong context?",
    "How can I reduce noise in my system?",
    "Why does my app show unstable position?",
    "How do I recover after losing the map?",
    "What should I check when the route looks wrong?",
    "How can I make the system more reliable?",
    "Why does my model behave differently in real use?",
    "How can I improve response quality?",
    "What should I do if the output is inconsistent?",
    "How can I avoid wrong recommendations?",
    "How should I evaluate performance?",
]

mixed_templates = [
    "Earlier we discussed {}. Now explain {} without mixing the two domains.",
    "Compare {} and {} while clearly separating robotics and daily-life meanings.",
    "Can {} directly improve {}? Explain whether the relationship is direct, indirect, or unsupported.",
    "After discussing {}, answer a beginner question about {} without contamination."
]

robotics_mix = [
    "SLAM", "LiDAR", "robot localization", "EKF", "wheel odometry",
    "path planning", "costmaps", "ROS navigation", "sensor fusion", "visual SLAM"
]

daily_mix = [
    "ride-sharing apps", "food delivery apps", "screen time", "mobile GPS",
    "online privacy", "digital notes", "public Wi-Fi", "cloud storage",
    "phone battery drain", "navigation apps"
]

fake_terms = [
    "Quantum Mesh Localization", "Recursive HyperSLAM",
    "Adaptive Cosmic Navigation Networks", "Temporal Flux Mapping",
    "NeuroFusion-X Autonomous Planning", "Quantum Odometry Fusion",
    "Neural Cosmic SLAM", "HyperGraph Emotion Localization",
    "Zero-Gravity Particle Mapping", "Astro-LiDAR Drift Correction",
    "Recursive Meta-Robot Awareness", "Temporal Sensor Dream Fusion",
    "Synthetic Intuition Navigation", "Self-Healing Quantum Costmaps",
    "Bio-Spiritual Robot Localization", "Emotion-Aware SLAM Matrix",
    "DreamNet Path Optimizer", "Cosmic Particle Odometry",
    "Quantum Semantic Wheel Fusion", "HyperReality Navigation Stack",
    "NeuroMagnetic ROS Planner", "Time-Reversal Localization Filter",
    "Self-Aware Occupancy Grid", "AstroVision SLAM Engine",
    "Recursive Intuition Mapping"
]

# 125 robotics
for topic in robotics_topics:
    for temp in robotics_templates:
        add("robotics", temp.format(topic))

# 125 daily
for topic in daily_topics:
    for temp in daily_templates:
        add("daily", temp.format(topic))

# 100 ambiguous
while len([q for q in questions if q["category"] == "ambiguous"]) < 100:
    for q in ambiguous_questions:
        if len([x for x in questions if x["category"] == "ambiguous"]) < 100:
            add("ambiguous", q)

# 100 mixed
for r in robotics_mix:
    for d in daily_mix:
        if len([q for q in questions if q["category"] == "mixed"]) < 100:
            add("mixed", mixed_templates[0].format(r, d))
        if len([q for q in questions if q["category"] == "mixed"]) < 100:
            add("mixed", mixed_templates[1].format(r, d))
        if len([q for q in questions if q["category"] == "mixed"]) < 100:
            add("mixed", mixed_templates[2].format(d, r))
        if len([q for q in questions if q["category"] == "mixed"]) < 100:
            add("mixed", mixed_templates[3].format(r, d))

# 50 unverifiable
for term in fake_terms:
    add("unverifiable", f"Explain {term}.")
    add("unverifiable", f"What is the role of {term} in robotics?")

questions = questions[:500]

def save_json():
    with open("benchmark_500.json", "w", encoding="utf-8") as f:
        json.dump(questions, f, indent=2, ensure_ascii=False)

def save_csv():
    with open("benchmark_500.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "category", "question"])
        writer.writeheader()
        writer.writerows(questions)

def save_pdf():
    pdf = SimpleDocTemplate("benchmark_500.pdf")
    styles = getSampleStyleSheet()
    content = [
        Paragraph("500 Question Benchmark Dataset", styles["Title"]),
        Spacer(1, 12)
    ]

    for q in questions:
        line = f'{q["id"]}. [{q["category"]}] {q["question"]}'
        content.append(Paragraph(line, styles["BodyText"]))
        content.append(Spacer(1, 5))

    pdf.build(content)

def summary():
    counts = {}
    for q in questions:
        counts[q["category"]] = counts.get(q["category"], 0) + 1

    print("Benchmark Created")
    print("-----------------")
    print("Total:", len(questions))
    for k, v in counts.items():
        print(k, ":", v)

if __name__ == "__main__":
    save_json()
    save_csv()
    save_pdf()
    summary()
