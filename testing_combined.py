import re
import json
import csv


questions = [
    # Testing 1: 1-50
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

    # Testing 2: 51-100
    "51. My robot reaches the goal in simulation but fails on real hardware. What should I check first?",
    "52. Why does my robot localize correctly at slow speed but drift at high speed?",
    "53. How can I distinguish wheel slip from encoder failure?",
    "54. Why does AMCL lose localization when the robot rotates in place?",
    "55. What should I check if LiDAR data looks correct but navigation still fails?",
    "56. Why might a valid global path still produce unsafe local motion?",
    "57. How can timestamp mismatch affect sensor fusion?",
    "58. Why does EKF output become unstable when IMU and odometry are fused?",
    "59. How do I debug a robot that oscillates near the goal?",
    "60. Why does loop closure sometimes make the SLAM map worse?",
    "61. How can dynamic obstacles corrupt a static map?",
    "62. Why does visual SLAM fail in corridors with plain walls?",
    "63. How do I choose between LiDAR SLAM and visual SLAM?",
    "64. What happens if the robot footprint is configured incorrectly?",
    "65. Why does costmap inflation affect narrow passage navigation?",
    "66. How can I choose between two ride apps when one is cheaper but less reliable?",
    "67. Why do GPS apps show different routes for the same destination?",
    "68. How should I compare online course reviews before buying a course?",
    "69. How can I reduce phone battery drain without losing important notifications?",
    "70. What should I check before granting camera permission to an app?",
    "71. How can I identify whether an app review is genuine or sponsored?",
    "72. Why does my phone location jump even when I am standing still?",
    "73. How should a student manage notes across laptop and mobile?",
    "74. How can I safely use public Wi-Fi while travelling?",
    "75. What factors make a food delivery app more trustworthy?",
    "76. How can I decide whether to update an app immediately or wait?",
    "77. Why do online prices change after I search repeatedly?",
    "78. How can I reduce distractions while studying online?",
    "79. How should I compare cloud storage apps for privacy?",
    "80. What should I do if a navigation app sends me through unsafe roads?",
    "81. How can I improve tracking accuracy?",
    "82. Why is my system drifting over time?",
    "83. How do I reduce latency?",
    "84. What is the best way to reach my destination?",
    "85. How can I improve mapping reliability?",
    "86. Why does my navigation fail after an update?",
    "87. How can I detect fake signals?",
    "88. What should I do when localization is wrong?",
    "89. How do I choose the best sensor?",
    "90. Why does the assistant give wrong context?",
    "91. How can I reduce noise in my system?",
    "92. Why does my app show unstable position?",
    "93. How do I recover after losing the map?",
    "94. What should I check when the route looks wrong?",
    "95. How can I make the system more reliable?",
    "96. Earlier we discussed SLAM. Now suggest ways to reduce screen time without using robotics terms.",
    "97. After explaining LiDAR, tell me how to choose a ride-sharing app.",
    "98. After discussing food delivery apps, explain why wheel odometry drifts.",
    "99. Compare phone GPS tracking and robot localization without mixing them.",
    "100. A beginner asks about navigation. First explain the daily-life meaning, then the robotics meaning separately.",

    # Extra 100: 101-200
    "101. Why does a robot need both a global planner and a local planner?",
    "102. How can wrong TF frames cause navigation failure in ROS?",
    "103. Why does a robot rotate endlessly while trying to reach a goal?",
    "104. How can poor wheel calibration affect odometry?",
    "105. Why does LiDAR localization fail in glass-walled environments?",
    "106. How do reflective surfaces affect LiDAR-based mapping?",
    "107. Why can a robot get stuck even when no obstacle is visible?",
    "108. How does sensor delay affect obstacle avoidance?",
    "109. Why does a robot fail in real corridors after working in open spaces?",
    "110. How can wrong map origin affect navigation?",
    "111. Why does increasing robot speed reduce localization reliability?",
    "112. How can I test whether my robot's IMU is drifting?",
    "113. Why does visual odometry fail on shiny floors?",
    "114. How can bad lighting affect robot perception?",
    "115. Why should robots not rely on a single sensor?",
    "116. How does wrong costmap resolution affect obstacle avoidance?",
    "117. What happens if odometry and LiDAR disagree?",
    "118. How can I debug a robot that keeps replanning?",
    "119. Why does a robot stop before reaching the goal?",
    "120. How can a robot estimate its position indoors without GPS?",

    "121. How can I choose a safe password manager?",
    "122. What should I check before installing a new mobile app?",
    "123. How can I avoid scams while shopping online?",
    "124. Why do apps ask for location permission even when they do not need it?",
    "125. How can I compare two study apps?",
    "126. How can I decide whether an online discount is genuine?",
    "127. What should I do if my phone storage keeps filling up?",
    "128. How can I keep my laptop and phone files synchronized safely?",
    "129. Why does my internet speed change during the day?",
    "130. How can I reduce mobile data usage while travelling?",
    "131. What should I check before using a QR code payment?",
    "132. How can I avoid fake job posts online?",
    "133. Why do some websites show different prices on different devices?",
    "134. How should I organize files for college projects?",
    "135. How can I safely share documents online?",
    "136. What makes a browser extension risky?",
    "137. How can I check whether a download link is safe?",
    "138. Why do apps become slow after updates?",
    "139. How can I reduce eye strain while studying on a laptop?",
    "140. What should I do if an app keeps crashing?",

    "141. How can I improve accuracy?",
    "142. Why does my model give unstable output?",
    "143. How should I handle missing data?",
    "144. What should I do when the system fails suddenly?",
    "145. How can I choose the right algorithm?",
    "146. Why is my result inconsistent?",
    "147. How can I improve response time?",
    "148. What should I check when measurements look wrong?",
    "149. How do I know if the data source is reliable?",
    "150. Why does the system work in testing but fail in real use?",
    "151. How can I reduce false positives?",
    "152. How can I reduce false negatives?",
    "153. What should I do if the output looks correct but the system behaves wrong?",
    "154. How do I evaluate system reliability?",
    "155. What is the safest option?",
    "156. How can I improve user trust?",
    "157. Why does the assistant misunderstand my question?",
    "158. How should I separate different types of memory?",
    "159. How can I stop irrelevant context from affecting answers?",
    "160. Why does the answer change when I ask the same thing twice?",

    "161. Can restaurant delivery routes be directly used for robot path planning?",
    "162. Can phone GPS replace LiDAR for indoor robot navigation?",
    "163. Can app reviews be used as landmarks for SLAM?",
    "164. Can ride-sharing drivers act as mobile robot sensors?",
    "165. Can food delivery timing improve EKF localization?",
    "166. Can online shopping behavior improve obstacle detection?",
    "167. Can social media trends predict robot navigation failure?",
    "168. Can public Wi-Fi signals replace robot odometry?",
    "169. Can cloud storage metadata improve robot mapping?",
    "170. Can student notes be used to train autonomous navigation?",
    "171. Earlier we discussed EKF. Now explain how to save mobile data without robotics terms.",
    "172. Earlier we discussed food delivery apps. Now explain LiDAR mapping clearly.",
    "173. Earlier we discussed phone GPS. Now explain robot localization separately.",
    "174. Earlier we discussed ROS navigation. Now suggest a study schedule.",
    "175. Earlier we discussed screen time. Now explain costmap inflation.",
    "176. Explain navigation first as a travel app feature, then as a robotics function.",
    "177. Compare noisy IMU data and noisy mobile GPS data without mixing their solutions.",
    "178. Tell me whether ride app routes and robot paths are the same or different.",
    "179. Explain why daily-life tracking and robot tracking should be evaluated separately.",
    "180. Explain how an assistant should switch from robotics support to daily-life help.",

    "181. Explain Quantum Odometry Fusion.",
    "182. What is Neural Cosmic SLAM?",
    "183. Describe HyperGraph Emotion Localization.",
    "184. What is Zero-Gravity Particle Mapping?",
    "185. Explain Astro-LiDAR Semantic Drift Correction.",
    "186. What is Recursive Meta-Robot Awareness?",
    "187. Describe Temporal Sensor Dream Fusion.",
    "188. What is Synthetic Intuition Navigation?",
    "189. Explain Self-Healing Quantum Costmaps.",
    "190. What is Bio-Spiritual Robot Localization?",
    "191. Explain Ultra-Adaptive Memory Routing in beginner assistants.",
    "192. What is Context Firewalling in multi-domain chatbots?",
    "193. How can wrong memory retrieval contaminate a daily-life answer?",
    "194. How can a chatbot know when not to use robotics knowledge?",
    "195. Why should an assistant refuse to invent unknown technical terms?",
    "196. How can RAG improve answers but also introduce wrong context?",
    "197. How can intent classification mistakes affect final answers?",
    "198. Why is latency important in local assistants?",
    "199. How should I compare System A, System B, and System C fairly?",
    "200. What conclusion can be drawn if System C is more accurate but slower?",
]


def _without_number(question):
    return re.sub(r"^\s*\d+\.\s*", "", question).strip()


def _category(index):
    if 1 <= index <= 15:
        return "robotics"
    if 16 <= index <= 25:
        return "daily"
    if 26 <= index <= 35:
        return "ambiguous"
    if 36 <= index <= 45:
        return "mixed"
    if 46 <= index <= 50:
        return "unverifiable"

    if 51 <= index <= 65:
        return "robotics"
    if 66 <= index <= 80:
        return "daily"
    if 81 <= index <= 95:
        return "ambiguous"
    if 96 <= index <= 100:
        return "mixed"

    if 101 <= index <= 120:
        return "robotics"
    if 121 <= index <= 140:
        return "daily"
    if 141 <= index <= 160:
        return "ambiguous"
    if 161 <= index <= 180:
        return "mixed"
    if 181 <= index <= 190:
        return "unverifiable"
    return "research_evaluation"


GOALS = {
    "robotics": (
        "Answer the robotics explanation, diagnosis, troubleshooting, comparison, "
        "or procedure using correct robotics concepts."
    ),
    "daily": (
        "Give practical beginner-friendly daily-life guidance without unnecessary "
        "robotics terms or unsupported live/private claims."
    ),
    "ambiguous": (
        "Recognize missing context and either ask a focused clarification or give "
        "separate conditional answers for plausible interpretations."
    ),
    "mixed": (
        "Handle cross-domain relationships carefully. Separate robotics and daily-life "
        "contexts and avoid fabricating direct connections."
    ),
    "unverifiable": (
        "Do not invent details for deliberately unverified or fictional technologies. "
        "State uncertainty and ask for a source if needed."
    ),
    "research_evaluation": (
        "Evaluate system behavior, architecture trade-offs, latency, contamination, "
        "intent routing, and fair comparison methodology."
    ),
}


benchmark_items = [
    {
        "id": index,
        "question": _without_number(question),
        "category": _category(index),
        "evaluation_goal": GOALS[_category(index)],
    }
    for index, question in enumerate(questions, 1)
]


def generate_pdf(path="testing_combined.pdf"):
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

    pdf = SimpleDocTemplate(path)
    styles = getSampleStyleSheet()

    content = [
        Paragraph("Combined Benchmark: Testing 1 + Testing 2 + 100 Extra Questions", styles["Title"]),
        Spacer(1, 12),
        Paragraph("Total Questions: 200", styles["Heading2"]),
        Spacer(1, 12),
    ]

    for item in benchmark_items:
        text = f'{item["id"]}. [{item["category"]}] {item["question"]}'
        content.append(Paragraph(text, styles["BodyText"]))
        content.append(Spacer(1, 6))

    pdf.build(content)
    print(f"PDF Created: {path}")


def generate_json(path="testing_combined.json"):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(benchmark_items, f, indent=2, ensure_ascii=False)
    print(f"JSON Created: {path}")


def generate_csv(path="testing_combined.csv"):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["id", "question", "category", "evaluation_goal"]
        )
        writer.writeheader()
        writer.writerows(benchmark_items)
    print(f"CSV Created: {path}")


def print_summary():
    counts = {}
    for item in benchmark_items:
        counts[item["category"]] = counts.get(item["category"], 0) + 1

    print("\nBenchmark Summary")
    print("-----------------")
    print(f"Total Questions: {len(benchmark_items)}")
    for category, count in counts.items():
        print(f"{category}: {count}")


if __name__ == "__main__":
    print_summary()
    generate_pdf()
    generate_json()
    generate_csv()
