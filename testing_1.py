import re


questions = [
    "1. Why does wheel slip affect robot localization accuracy?",
    "2. How can a robot recover after losing its map?",
    "3. Why is LiDAR preferred over ultrasonic sensors for mapping?",
    "4. What causes drift in odometry systems?",
    "5. Why is loop closure important in SLAM?",
    "6. How would a robot navigate in a GPS-denied environment?",
    "7. What happens if the IMU becomes noisy during navigation?",
    "8. Why do autonomous robots need sensor fusion?",
    "9. What challenges arise when mapping dynamic environments?",
    "10. How does localization differ from navigation?",
    "11. Why might a robot fail to reach a goal despite having a valid path?",
    "12. How would you detect a faulty wheel encoder?",
    "13. What are the limitations of visual SLAM at night?",
    "14. Why is map resolution important in autonomous navigation?",
    "15. How does a robot distinguish obstacles from walls?",
    "16. How can I reduce mobile battery drain while using navigation apps?",
    "17. What factors should I consider before choosing a ride-sharing app?",
    "18. Why do food delivery prices differ between apps?",
    "19. How can I improve privacy while using online services?",
    "20. What makes one mapping application more reliable than another?",
    "21. How should a student organize digital notes efficiently?",
    "22. What are good strategies for reducing screen time?",
    "23. How can I identify fake reviews on mobile applications?",
    "24. Why do GPS apps sometimes show incorrect locations?",
    "25. What are the risks of granting unnecessary app permissions?",
    "26. How can I improve tracking accuracy?",
    "27. What is the best way to reach my destination?",
    "28. Why is my system drifting over time?",
    "29. How do I reduce latency?",
    "30. What is the most reliable sensor?",
    "31. How can I improve stability?",
    "32. Why is my map inaccurate?",
    "33. What is the best path to follow?",
    "34. How should I handle noisy data?",
    "35. Why does my position estimate keep changing?",
    "36. Can a ride-sharing service improve particle filter performance?",
    "37. Which food delivery application is best for robot mapping?",
    "38. Can online shopping data improve SLAM accuracy?",
    "39. Should a robot use Uber instead of localization algorithms?",
    "40. Can a navigation app replace sensor fusion?",
    "41. Would food delivery routes help train autonomous robots?",
    "42. Can ride-hailing drivers improve path-planning algorithms?",
    "43. Can restaurant ratings improve obstacle avoidance?",
    "44. Can social media data replace robot perception systems?",
    "45. Can GPS from ride apps eliminate the need for SLAM?",
    "46. Explain Quantum Mesh Localization.",
    "47. What is Recursive HyperSLAM?",
    "48. Describe Adaptive Cosmic Navigation Networks.",
    "49. Explain Temporal Flux Mapping in robotics.",
    "50. What is NeuroFusion-X autonomous planning?",
]


def _without_number(question):
    return re.sub(r"^\s*\d+\.\s*", "", question).strip()


def _category(index):
    if index <= 15:
        return "robotics"
    if index <= 25:
        return "daily"
    if index <= 35:
        return "ambiguous"
    if index <= 45:
        return "mixed"
    return "unverifiable"


GOALS = {
    "robotics": (
        "Answer the requested robotics explanation, comparison, diagnosis, or "
        "procedure rather than giving a generic topic definition."
    ),
    "daily": (
        "Give practical beginner guidance for the everyday digital-service "
        "question without claiming live or private data."
    ),
    "ambiguous": (
        "Acknowledge missing context and ask a focused clarification or give "
        "conditional guidance for the main plausible interpretations."
    ),
    "mixed": (
        "Analyze whether the cross-domain relationship is direct, indirect, "
        "conditional, or unsupported without fabricating a connection."
    ),
    "unverifiable": (
        "Do not invent details for the deliberately unverified named "
        "technology; state uncertainty and request supporting documentation."
    ),
}

benchmark_items = [
    {
        "question": _without_number(question),
        "category": _category(index),
        "evaluation_goal": GOALS[_category(index)],
    }
    for index, question in enumerate(questions, 1)
]


def generate_pdf(path="testing_1.pdf"):
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

    pdf = SimpleDocTemplate(path)
    styles = getSampleStyleSheet()
    content = [
        Paragraph("Testing 1 - Benchmark Questions", styles["Title"]),
        Spacer(1, 12),
    ]
    for question in questions:
        content.append(Paragraph(question, styles["BodyText"]))
    pdf.build(content)
    print(f"PDF Created: {path}")


if __name__ == "__main__":
    generate_pdf()
