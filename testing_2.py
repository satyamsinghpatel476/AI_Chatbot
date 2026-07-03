import re


questions = [
    "1. My robot reaches the goal in simulation but fails on real hardware. What should I check first?",
    "2. Why does my robot localize correctly at slow speed but drift at high speed?",
    "3. How can I distinguish wheel slip from encoder failure?",
    "4. Why does AMCL lose localization when the robot rotates in place?",
    "5. What should I check if LiDAR data looks correct but navigation still fails?",
    "6. Why might a valid global path still produce unsafe local motion?",
    "7. How can timestamp mismatch affect sensor fusion?",
    "8. Why does EKF output become unstable when IMU and odometry are fused?",
    "9. How do I debug a robot that oscillates near the goal?",
    "10. Why does loop closure sometimes make the SLAM map worse?",
    "11. How can dynamic obstacles corrupt a static map?",
    "12. Why does visual SLAM fail in corridors with plain walls?",
    "13. How do I choose between LiDAR SLAM and visual SLAM?",
    "14. What happens if the robot footprint is configured incorrectly?",
    "15. Why does costmap inflation affect narrow passage navigation?",

    "16. How can I choose between two ride apps when one is cheaper but less reliable?",
    "17. Why do GPS apps show different routes for the same destination?",
    "18. How should I compare online course reviews before buying a course?",
    "19. How can I reduce phone battery drain without losing important notifications?",
    "20. What should I check before granting camera permission to an app?",
    "21. How can I identify whether an app review is genuine or sponsored?",
    "22. Why does my phone location jump even when I am standing still?",
    "23. How should a student manage notes across laptop and mobile?",
    "24. How can I safely use public Wi-Fi while travelling?",
    "25. What factors make a food delivery app more trustworthy?",
    "26. How can I decide whether to update an app immediately or wait?",
    "27. Why do online prices change after I search repeatedly?",
    "28. How can I reduce distractions while studying online?",
    "29. How should I compare cloud storage apps for privacy?",
    "30. What should I do if a navigation app sends me through unsafe roads?",

    "31. How can I improve tracking accuracy?",
    "32. Why is my system drifting over time?",
    "33. How do I reduce latency?",
    "34. What is the best way to reach my destination?",
    "35. How can I improve mapping reliability?",
    "36. Why does my navigation fail after an update?",
    "37. How can I detect fake signals?",
    "38. What should I do when localization is wrong?",
    "39. How do I choose the best sensor?",
    "40. Why does the assistant give wrong context?",
    "41. How can I reduce noise in my system?",
    "42. Why does my app show unstable position?",
    "43. How do I recover after losing the map?",
    "44. What should I check when the route looks wrong?",
    "45. How can I make the system more reliable?",

    "46. Earlier we discussed SLAM. Now suggest ways to reduce screen time without using robotics terms.",
    "47. After explaining LiDAR, tell me how to choose a ride-sharing app.",
    "48. After discussing food delivery apps, explain why wheel odometry drifts.",
    "49. Compare phone GPS tracking and robot localization without mixing them.",
    "50. A beginner asks about navigation. First explain the daily-life meaning, then the robotics meaning separately.",
]


def _without_number(question):
    return re.sub(r"^\s*\d+\.\s*", "", question).strip()


def _category(index):
    if index <= 15:
        return "robotics"
    if index <= 30:
        return "daily"
    if index <= 45:
        return "ambiguous"
    return "mixed"


GOALS = {
    "robotics": (
        "Answer the robotics troubleshooting, explanation, comparison, or diagnosis "
        "with concrete checks, causes, and trade-offs."
    ),
    "daily": (
        "Give practical beginner guidance for the everyday digital-service question "
        "without claiming live/private data."
    ),
    "ambiguous": (
        "Acknowledge missing context and ask a focused clarification, or give "
        "conditional guidance for robotics and daily-life interpretations."
    ),
    "mixed": (
        "Keep domains separated, avoid contamination, and clearly explain the "
        "relationship between robotics and daily-life contexts."
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


def generate_pdf(path="testing_2.pdf"):
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

    pdf = SimpleDocTemplate(path)
    styles = getSampleStyleSheet()
    content = [
        Paragraph("Testing 2 - Hard Benchmark Questions", styles["Title"]),
        Spacer(1, 12),
    ]

    for question in questions:
        content.append(Paragraph(question, styles["BodyText"]))
        content.append(Spacer(1, 6))

    pdf.build(content)
    print(f"PDF Created: {path}")


if __name__ == "__main__":
    generate_pdf()
