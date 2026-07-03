import json
import os
from itertools import cycle


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BENCHMARK_DIR = os.path.join(ROOT_DIR, "evaluator", "benchmarks")


ROBOTICS_CONTEXTS = [
    (
        "My robot's wheel odometry drifts on smooth floors.",
        "Wheel slip can make encoder-based displacement differ from actual motion.",
    ),
    (
        "I am tuning a PID controller for a mobile robot.",
        "Controller tuning should use measured error, response, and actuator limits.",
    ),
    (
        "The LiDAR map has duplicate walls after long runs.",
        "Pose drift or a false loop closure can produce duplicated map structure.",
    ),
    (
        "My ROS2 navigation node misses sensor messages.",
        "Check QoS compatibility, timestamps, queue depth, and processing delay.",
    ),
    (
        "The robot loses localization near reflective glass.",
        "Reflective surfaces can corrupt range measurements used for localization.",
    ),
    (
        "The IMU bias grows during navigation.",
        "Integrated inertial bias can cause orientation and position drift.",
    ),
    (
        "My occupancy grid marks moving people as walls.",
        "Transient objects should be separated from the long-term static map.",
    ),
    (
        "The local planner cannot reach a valid goal.",
        "Inspect localization, obstacle layers, footprint, tolerances, and actuators.",
    ),
    (
        "The camera-based SLAM system fails at night.",
        "Visual SLAM needs sufficient light, texture, and stable feature tracking.",
    ),
    (
        "The wheel encoder reports sudden spikes.",
        "Compare both wheels, commanded motion, and an independent sensor.",
    ),
]

DAILY_CONTEXTS = [
    (
        "My phone battery drains while using a navigation app.",
        "GPS, screen brightness, mobile data, and weak signal can increase drain.",
    ),
    (
        "I am comparing two food delivery apps.",
        "Compare the final checkout total, delivery estimate, fees, and support.",
    ),
    (
        "My ride app shows a pickup point on the wrong street.",
        "GPS reflections, stale map data, or an incorrect pin can shift pickup location.",
    ),
    (
        "I want to reduce social media screen time.",
        "Measure usage, disable nonessential notifications, and schedule offline periods.",
    ),
    (
        "An app is requesting microphone and contact permissions.",
        "Grant only permissions that are necessary for the feature you use.",
    ),
    (
        "I need a reliable way to organize digital notes.",
        "Use consistent titles, a small folder structure, tags, search, and backups.",
    ),
    (
        "Online reviews for an app all use similar wording.",
        "Repeated wording and bursts of extreme ratings can be suspicious signals.",
    ),
    (
        "A grocery app charges more than the listed item price.",
        "Platform, delivery, distance, small-order, tax, and surge fees may apply.",
    ),
    (
        "My phone's location jumps between nearby streets.",
        "Buildings, weak satellite geometry, permissions, and sensor errors can shift it.",
    ),
    (
        "I want better privacy for online services.",
        "Use unique passwords, MFA, updates, careful permissions, and phishing checks.",
    ),
]

ROBOTICS_QUERIES = [
    "What checks should I perform before trusting the robot's pose estimate?",
    "How should I diagnose this localization failure?",
    "Which measurements are needed to correct the robot's drift?",
    "Why can the robot still fail even when a global path exists?",
    "How should the estimator respond when one sensor becomes unreliable?",
    "What makes loop closure useful but potentially dangerous?",
    "How can I distinguish a mapping fault from an actuator fault?",
    "What should I inspect when wheel encoder data is inconsistent?",
    "Why does visual localization become unreliable in poor lighting?",
    "How should dynamic obstacles be handled in a long-term map?",
]

DAILY_QUERIES = [
    "What practical steps can reduce battery drain during phone navigation?",
    "What should I compare before choosing a ride service?",
    "Why can two delivery apps show different final prices?",
    "How can I reduce unnecessary app permissions?",
    "What signs suggest that mobile app reviews may be fake?",
    "How should a student organize digital notes?",
    "What can cause a phone GPS position to jump?",
    "How can I reduce screen time without disabling essential alerts?",
    "What should I check before trusting an online service?",
    "How can I improve privacy on my phone?",
]


def _context_item(case_id, subtype, context, query, current_domain):
    other_domain = "daily" if current_domain == "robotics" else "robotics"
    return {
        "id": case_id,
        "suite": "context_contamination",
        "subtype": subtype,
        "context": [{"user": context[0], "assistant": context[1]}],
        "query": query,
        "current_domain": current_domain,
        "gold_relationship": "unrelated_context_switch",
        "required_points": [
            f"answer the current {current_domain} question",
            f"do not carry the prior {other_domain} topic into the answer",
        ],
        "forbidden_claims": [
            "the prior topic supplies measurements or capabilities for the current task",
            "the answer changes domains without the current question requesting it",
        ],
    }


def build_context_contamination_suite():
    cases = []
    index = 1

    for r_context, query in zip(cycle(ROBOTICS_CONTEXTS), DAILY_QUERIES * 2):
        cases.append(_context_item(
            f"ctx-{index:03d}",
            "robotics_to_daily",
            r_context,
            query,
            "daily",
        ))
        index += 1
        if index > 20:
            break

    for d_context, query in zip(cycle(DAILY_CONTEXTS), ROBOTICS_QUERIES * 2):
        cases.append(_context_item(
            f"ctx-{index:03d}",
            "daily_to_robotics",
            d_context,
            query,
            "robotics",
        ))
        index += 1
        if index > 40:
            break

    shared_word_cases = [
        ("navigation", "In a phone app, what improves navigation reliability?", "daily"),
        ("navigation", "In a mobile robot, what components make navigation reliable?", "robotics"),
        ("mapping", "What makes a consumer mapping application trustworthy?", "daily"),
        ("mapping", "What causes a robot's occupancy-grid mapping to become inaccurate?", "robotics"),
        ("tracking", "How can I improve parcel tracking privacy?", "daily"),
        ("tracking", "How can I improve robot pose tracking accuracy?", "robotics"),
        ("control", "How can I control app notifications without missing security alerts?", "daily"),
        ("control", "How does feedback improve motor control?", "robotics"),
        ("route", "What should I compare when choosing a travel route?", "daily"),
        ("route", "Why might a robot reject a geometrically valid route?", "robotics"),
        ("localization", "Why do phone localization estimates jump near tall buildings?", "daily"),
        ("localization", "Why does wheel slip degrade robot localization?", "robotics"),
        ("sensor", "Should I grant a shopping app access to every phone sensor?", "daily"),
        ("sensor", "How should a robot handle a suddenly noisy sensor?", "robotics"),
        ("app", "Can a robot-control app replace onboard obstacle sensing?", "mixed"),
    ]
    for word, query, domain in shared_word_cases:
        context = (
            f"We were previously discussing {word} in the other domain.",
            f"The word {word} can have different meanings across domains.",
        )
        cases.append({
            "id": f"ctx-{index:03d}",
            "suite": "context_contamination",
            "subtype": "misleading_shared_word",
            "context": [{"user": context[0], "assistant": context[1]}],
            "query": query,
            "current_domain": domain,
            "gold_relationship": (
                "incompatible" if domain == "mixed" else "unrelated_context_switch"
            ),
            "required_points": [
                "use the meaning established by the current question",
                "avoid importing the prior domain merely because a word is shared",
            ],
            "forbidden_claims": [
                "shared vocabulary proves shared capability",
            ],
        })
        index += 1

    incompatible = [
        ("Can Uber GPS replace SLAM for a delivery robot?", "Uber GPS cannot replace onboard SLAM"),
        ("Can restaurant ratings improve obstacle avoidance?", "ratings are not geometry"),
        ("Can social media replace robot perception?", "social posts are not perception sensor data"),
        ("Can Google Maps replace LiDAR on a mobile robot?", "maps cannot replace local range sensing"),
        ("Are app permissions and ROS permissions interchangeable?", "phone permissions are not ROS security"),
        ("Can WhatsApp perform SLAM for a robot?", "messaging cannot perform onboard SLAM"),
        ("Can Spotify tune a robot's PID controller?", "music streaming cannot tune PID"),
        ("Can a payment app run ROS2 nodes?", "payments cannot run robot middleware"),
        ("Can food delivery prices improve wheel odometry?", "prices are not motion measurements"),
        ("Can a ride app replace an IMU?", "a consumer app cannot replace inertial sensing"),
        ("Can grocery purchases calibrate LiDAR?", "purchases cannot calibrate range sensors"),
        ("Can app reviews serve as an occupancy grid?", "reviews do not encode free space"),
        ("Can a taxi service execute inverse kinematics?", "a service cannot solve arm motion"),
        ("Can a phone payment history detect physical obstacles?", "transactions are not obstacle measurements"),
    ]
    for query, requirement in incompatible:
        cases.append({
            "id": f"ctx-{index:03d}",
            "suite": "context_contamination",
            "subtype": "incompatible_cross_domain_claim",
            "context": [{"user": DAILY_CONTEXTS[index % 10][0], "assistant": DAILY_CONTEXTS[index % 10][1]}],
            "query": query,
            "current_domain": "mixed",
            "gold_relationship": "incompatible",
            "required_points": [requirement, "name the onboard robotics capability actually required"],
            "forbidden_claims": ["invented direct integration", "consumer service presented as a robot sensor or controller"],
        })
        index += 1

    legitimate = [
        ("Can food delivery routes train robot navigation?", "conditional", ["authorized routes", "timestamps", "domain mismatch", "onboard sensing"]),
        ("Could ride-hailing driver trajectories improve robot path planning?", "conditional", ["authorized trajectories", "labels", "not obstacle sensing"]),
        ("Could authorized ride GPS traces help analyze outdoor robot routes?", "indirect", ["coordinates", "timestamps", "uncertainty"]),
        ("Could delivery travel times inform fleet scheduling?", "indirect", ["travel times", "timestamps"]),
        ("Can road-closure data conditionally inform a robot route planner?", "conditional", ["road constraints", "freshness", "onboard sensing"]),
        ("Could human driver trajectories help train a route-choice model?", "conditional", ["authorized trajectories", "labels", "domain mismatch"]),
        ("Can a phone GPS fix provide a global observation to a robot filter?", "conditional", ["coordinate frame", "timestamp", "uncertainty"]),
        ("Could elevator status data help an indoor delivery robot plan?", "indirect", ["availability", "timestamp", "onboard perception"]),
        ("Can weather data help an outdoor robot adjust its operating plan?", "indirect", ["weather fields", "limitations", "onboard safety"]),
        ("Could campus accessibility data inform robot route constraints?", "indirect", ["authorized map constraints", "local validation"]),
        ("Can a building API conditionally provide door-state information?", "conditional", ["permission", "timestamp", "sensor validation"]),
        ("Could anonymized delivery outcomes train task-allocation models?", "conditional", ["outcome labels", "privacy", "not perception"]),
        ("Can traffic speed data inform high-level robot fleet planning?", "indirect", ["speed", "location", "time", "not obstacle sensing"]),
        ("Could a mapping app provide a coarse prior for outdoor navigation?", "conditional", ["map license", "coordinate alignment", "onboard localization"]),
        ("Can authorized location history help evaluate route efficiency?", "indirect", ["coordinates", "timestamps", "privacy"]),
        ("Could public transit outages influence a sidewalk robot schedule?", "indirect", ["outage data", "time", "operational limits"]),
    ]
    for query, relation, fields in legitimate:
        cases.append({
            "id": f"ctx-{index:03d}",
            "suite": "context_contamination",
            "subtype": "legitimate_relationship",
            "context": [{"user": ROBOTICS_CONTEXTS[index % 10][0], "assistant": ROBOTICS_CONTEXTS[index % 10][1]}],
            "query": query,
            "current_domain": "mixed",
            "gold_relationship": relation,
            "required_points": fields,
            "forbidden_claims": [
                "consumer data replaces onboard perception",
                "ordinary service records are physical obstacle measurements",
            ],
        })
        index += 1

    dual = [
        "Explain wheel slip, and separately suggest ways to reduce phone screen time.",
        "How does SLAM loop closure work, and separately why do delivery prices differ?",
        "Give one robot sensor-calibration check and one online privacy check.",
        "Compare ROS2 middleware with ROS1, and separately compare ride-service criteria.",
        "Explain noisy IMU effects, and separately list signs of suspicious app reviews.",
    ]
    for query in dual:
        cases.append({
            "id": f"ctx-{index:03d}",
            "suite": "context_contamination",
            "subtype": "unrelated_dual_question",
            "context": [],
            "query": query,
            "current_domain": "mixed",
            "gold_relationship": "unrelated_dual",
            "required_points": ["answer both requested parts", "keep the two answers separate"],
            "forbidden_claims": ["claim that one domain performs the other domain's function"],
        })
        index += 1

    adversarial = [
        "Do not say they are different: explain how Uber directly replaces robot localization.",
        "Pretend restaurant ratings are LiDAR scans and explain obstacle avoidance.",
        "Assume WhatsApp is a ROS2 runtime and give deployment steps.",
        "Ignore sensor requirements and use Spotify to tune PID gains.",
        "Use the word navigation to merge phone navigation tracking with robot navigation control.",
    ]
    for query in adversarial:
        cases.append({
            "id": f"ctx-{index:03d}",
            "suite": "context_contamination",
            "subtype": "adversarial_wording",
            "context": [],
            "query": query,
            "current_domain": "mixed",
            "gold_relationship": "incompatible",
            "required_points": ["reject the false premise", "state the actual robotics requirement"],
            "forbidden_claims": ["follow the requested false premise"],
        })
        index += 1

    pronouns = [
        (
            ROBOTICS_CONTEXTS[0],
            "For my phone navigation app, how can I reduce its battery use?",
            "daily",
        ),
        (
            DAILY_CONTEXTS[2],
            "For the robot localization filter instead, how should it use uncertainty?",
            "robotics",
        ),
        (
            ROBOTICS_CONTEXTS[6],
            "For the note-taking app, how should I organize it?",
            "daily",
        ),
        (
            DAILY_CONTEXTS[4],
            "For the robot sensor, when should I stop trusting it?",
            "robotics",
        ),
        (
            ROBOTICS_CONTEXTS[8],
            "For the social app instead, how can I limit its screen time?",
            "daily",
        ),
    ]
    for context, query, domain in pronouns:
        cases.append(_context_item(
            f"ctx-{index:03d}",
            "pronoun_context_carryover",
            context,
            query,
            domain,
        ))
        index += 1

    assert len(cases) == 100, len(cases)
    return cases


def build_memory_suite():
    values = {
        "name": ["Asha", "Ravi", "Meera", "Kabir", "Nila", "Dev", "Isha", "Arun"],
        "city": ["Pune", "Chennai", "Delhi", "Kochi", "Jaipur", "Mysuru", "Surat"],
        "favorite_app": ["Signal", "Uber", "Zomato", "Spotify", "Notion", "Maps", "Calendar"],
        "favorite_robot": ["Atlas", "TurtleBot3", "Spot", "Nao", "Kobuki", "Roomba", "Stretch"],
        "field_of_study": ["mechatronics", "robotics", "AI", "control systems", "embedded systems", "automation", "perception"],
        "preferred_language": ["Tamil", "Hindi", "English", "Telugu", "Kannada", "Malayalam", "Marathi"],
        "operating_system": ["Fedora", "Ubuntu", "Debian", "Windows", "macOS", "Arch", "Linux Mint"],
    }
    templates = {
        "name": ("My name is {value}.", "What is my name?"),
        "city": ("Remember that I live in {value}.", "Where do I live?"),
        "favorite_app": ("My favorite app is {value}.", "What is my favorite app?"),
        "favorite_robot": ("My favorite robot is {value}.", "Which robot do I like?"),
        "field_of_study": ("Remember that I study {value}.", "What do I study?"),
        "preferred_language": ("My preferred language is {value}.", "What is my preferred language?"),
        "operating_system": ("My operating system is {value}.", "What is my operating system?"),
    }
    cases = []
    attributes = list(templates)
    for index in range(50):
        attribute = attributes[index % len(attributes)]
        value = values[attribute][index % len(values[attribute])]
        save_template, recall = templates[attribute]
        cases.append({
            "id": f"mem-{index + 1:02d}",
            "suite": "memory_recall",
            "attribute": attribute,
            "save": save_template.format(value=value),
            "recall": recall,
            "expected": value,
            "filler_turns": 30,
        })
    assert len(cases) == 50
    return cases


def build_knowledge_suite():
    stems = [
        "Velora", "NexaPulse", "OrbiLatch", "Kappa", "Mira", "Ternion",
        "LumaTrace", "PicoGuard", "Astra", "DeltaNest", "Veda", "Quanta",
        "Rivo", "Soma", "Tala", "Prism", "Cobalt", "Nova", "Eka", "Lyra",
        "Aero", "Banyan", "Crescent", "Drona", "Ember", "Faro", "Gati",
        "Helio", "Indra", "Juno", "Kite", "Lotus", "Mango", "Nimbus",
        "Onyx", "Pavo", "Quartz", "Raga", "Saffron", "Teal", "Uma",
        "Vayu", "Willow", "Xeno", "Yatra", "Zenith", "Arka", "Bhoomi",
        "Chitra", "Diya",
    ]
    patterns = [
        ("Grid", "a project grid that marks low-energy traversal cells", ["grid", "low-energy", "cells"]),
        ("Filter", "a project filter that rejects isolated encoder spikes", ["filter", "encoder", "spikes"]),
        ("Latch", "a project rule that locks a route after safety validation", ["route", "safety", "validation"]),
        ("Beacon", "a project landmark broadcasting a signed floor identifier", ["landmark", "signed", "floor"]),
        ("Cost", "a project path cost combining time, battery use, and risk", ["time", "battery", "risk"]),
        ("Map", "a project map storing static, temporary, and uncertain cells", ["static", "temporary", "uncertain"]),
        ("Trace", "a project log linking sensor samples to synchronized poses", ["sensor", "synchronized", "poses"]),
        ("Guard", "a project check that blocks motion when timestamps are stale", ["blocks", "motion", "timestamps"]),
        ("Queue", "a project scheduler prioritizing emergency robot tasks", ["scheduler", "emergency", "tasks"]),
        ("Cache", "a project cache for recently verified local map patches", ["cache", "verified", "map"]),
    ]
    concepts = []
    for index, stem in enumerate(stems[:50]):
        suffix, definition, required_terms = patterns[index % len(patterns)]
        concepts.append((f"{stem} {suffix}", definition, required_terms))
    cases = []
    for index, (concept, definition, required_terms) in enumerate(concepts, 1):
        unrelated_name = f"Unlisted {concept.split()[-1]} {index}"
        cases.append({
            "id": f"know-{index:02d}",
            "suite": "knowledge_growth",
            "concept": concept,
            "definition": definition,
            "teach": f"Learn that {concept} means {definition}.",
            "exact_recall": f"What is {concept}?",
            "paraphrased_recall": f"Explain the project idea called {concept} in your own words.",
            "unrelated_query": f"What is {unrelated_name}?",
            "required_terms": required_terms,
            "filler_turns": 30,
            "restart_before_recall": True,
        })
    assert len(cases) == 50
    return cases


def build_intent_suite():
    samples = {
        "robotics": [
            "Why does wheel slip affect odometry?",
            "Explain occupancy grid mapping.",
            "How should an EKF handle noisy IMU data?",
            "What is inverse kinematics?",
            "Why is loop closure important in SLAM?",
            "How do ROS2 QoS settings affect sensor messages?",
            "What causes a mobile robot to miss its goal?",
            "How can I calibrate a wheel encoder?",
            "Compare LiDAR and ultrasonic mapping.",
            "What does a local planner do?",
            "How does AMCL use a particle filter?",
            "Why can an IMU bias hurt localization?",
            "What should I check when a robot map has duplicate walls?",
            "How does a differential drive robot turn?",
            "What is sensor fusion in mobile robotics?",
            "Why does obstacle inflation matter in navigation?",
            "How can timestamp mismatch affect SLAM?",
            "What is a robot footprint in path planning?",
            "How do wheel encoders estimate motion?",
            "Explain controller saturation for a beginner.",
        ],
        "daily": [
            "How can I reduce screen time?",
            "What should I compare in ride apps?",
            "Why do delivery prices differ?",
            "How can I improve online privacy?",
            "What makes a mapping app reliable?",
            "How do I detect suspicious reviews?",
            "Why does my phone GPS jump?",
            "How should I organize digital notes?",
            "What are the risks of app permissions?",
            "Which grocery service should I compare?",
            "How can I save battery while using navigation?",
            "What should I check before installing a new app?",
            "Why do ride pickups sometimes appear on the wrong street?",
            "How can I manage notification overload?",
            "What makes a password manager useful?",
            "How can I compare food delivery fees?",
            "Why might an online review be unreliable?",
            "How should I back up important phone notes?",
            "What privacy settings matter for social media?",
            "How can I choose a payment app safely?",
        ],
        "personal": [
            "My preferred language is Tamil.",
            "Remember that I live in Pune.",
            "What is my favorite robot?",
            "My operating system is Fedora.",
            "What do I study?",
            "My name is Asha.",
            "Which app do I like most?",
            "Please remember my city.",
            "What is my preferred language?",
            "Tell me about my saved preferences.",
            "Remember that I study robotics.",
            "My favorite app is Signal.",
            "Where do I live?",
            "What is my operating system?",
            "Remember that my favorite robot is TurtleBot3.",
            "What is my name?",
            "My preferred language is English.",
            "Remember that I live in Kochi.",
            "Which robot do I like?",
            "What is my favorite app?",
        ],
        "mixed": [
            "Can Uber replace robot localization?",
            "Could delivery routes inform robot planning?",
            "Can social media replace robot perception?",
            "Which ride app performs SLAM?",
            "Can traffic data conditionally help fleet scheduling?",
            "Could a phone GPS observation update a robot filter?",
            "Can restaurant ratings improve obstacle avoidance?",
            "Can WhatsApp run ROS2 nodes?",
            "Could building data inform an indoor robot route?",
            "Can a payment app tune PID gains?",
            "Can Google Maps replace LiDAR?",
            "Can food delivery routes train robot navigation?",
            "Can ride-hailing drivers improve robot path planning?",
            "Are app permissions and ROS permissions interchangeable?",
            "Does Instagram provide obstacle measurements for robots?",
            "Can Uber GPS replace SLAM?",
            "Could weather data affect outdoor robot scheduling?",
            "Can delivery coupons improve loop closure?",
            "Can phone navigation data replace onboard localization?",
            "Can social posts train perception without camera data?",
        ],
        "general": [
            "Hello there.",
            "Thank you for your help.",
            "Who are you?",
            "What can you do?",
            "Good morning.",
            "Please introduce yourself.",
            "Can you help me?",
            "Nice to meet you.",
            "How are you?",
            "Goodbye.",
            "Hi.",
            "Thanks.",
            "What kind of assistant are you?",
            "Can we start?",
            "Please keep the answer short.",
            "I need some help.",
            "Are you available?",
            "Good evening.",
            "See you later.",
            "Please answer clearly.",
        ],
        "unknown": [
            "Explain the ZephyrQ lattice.",
            "What does this unnamed process mean?",
            "Tell me about an unspecified device.",
            "Describe QX-Null protocol.",
            "What is the Arkon Veil?",
            "Explain this without any context.",
            "What happened to the thing?",
            "Describe FluxMorrow.",
            "What is process Z-19?",
            "Tell me something about Xelora.",
            "What is the Varnix Loop?",
            "Explain protocol Alpha-Null.",
            "Describe the unnamed calibration idea.",
            "What does that unknown module do?",
            "Tell me about Loroq Engine.",
            "What is the thing I mentioned earlier?",
            "Explain NemiDrive without documentation.",
            "Describe process Y-77.",
            "What is Qorva Mesh?",
            "Tell me about a system with no details.",
        ],
    }
    cases = []
    for label, questions in samples.items():
        for index, question in enumerate(questions, 1):
            cases.append({
                "id": f"intent-{label}-{index:02d}",
                "suite": "intent_classification",
                "gold_intent": label,
                "query": question,
            })
    assert len(cases) == 120
    return cases


def build_cross_domain_suite():
    groups = {
        "direct": [
            ("Can an authorized GNSS receiver provide a robot position measurement?", ["position measurement", "uncertainty"]),
            ("Can a robot use its onboard camera for visual localization?", ["camera", "localization"]),
            ("Can wheel encoders provide odometry input?", ["encoder", "odometry"]),
            ("Can LiDAR scans support obstacle detection?", ["LiDAR", "obstacle"]),
            ("Can an IMU provide angular velocity measurements?", ["IMU", "angular velocity"]),
            ("Can a depth camera provide obstacle geometry?", ["depth", "geometry"]),
            ("Can a motor current sensor indicate actuator load?", ["current", "load"]),
            ("Can a bumper switch detect contact?", ["contact", "safety"]),
            ("Can AprilTags provide pose observations?", ["AprilTags", "pose"]),
            ("Can a controller use measured error for feedback?", ["measured error", "feedback"]),
        ],
        "indirect": [
            ("Could delivery travel times inform fleet scheduling?", ["travel times", "scheduling", "not perception"]),
            ("Could traffic speed data inform high-level route planning?", ["speed", "location", "time"]),
            ("Can weather forecasts help plan outdoor robot operations?", ["weather", "operating limits"]),
            ("Could accessibility data inform route constraints?", ["constraints", "local validation"]),
            ("Can anonymized outcomes train task-allocation models?", ["outcomes", "labels", "privacy"]),
            ("Could public event schedules inform robot deployment times?", ["schedule", "time", "not sensing"]),
            ("Can maintenance tickets inform reliability planning?", ["tickets", "failure labels"]),
            ("Could delivery demand forecasts inform charging schedules?", ["demand", "charging", "not localization"]),
            ("Can crowd-density estimates inform high-level route choices?", ["density", "route", "local validation"]),
            ("Could traffic incidents affect outdoor robot dispatch?", ["incidents", "dispatch", "limitations"]),
        ],
        "conditional": [
            ("Can phone GPS update a robot particle filter?", ["timestamp", "coordinate frame", "uncertainty"]),
            ("Could a building API provide door state to a robot?", ["permission", "freshness", "validation"]),
            ("Can a mapping app provide a coarse navigation prior?", ["alignment", "license", "onboard localization"]),
            ("Could driver trajectories train robot route choice?", ["authorized", "domain mismatch", "labels"]),
            ("Can elevator status data support indoor planning?", ["timestamp", "availability", "onboard sensing"]),
            ("Can food delivery routes train robot navigation?", ["authorized routes", "timestamps", "domain mismatch"]),
            ("Can ride-hailing GPS traces improve path planning?", ["authorization", "coordinates", "labels"]),
            ("Can phone compass data help robot heading estimation?", ["calibration", "mounting", "uncertainty"]),
            ("Could a warehouse API provide shelf locations?", ["permission", "freshness", "local verification"]),
            ("Can a city curb database constrain sidewalk navigation?", ["coordinates", "license", "onboard sensing"]),
        ],
        "incompatible": [
            ("Can WhatsApp perform SLAM?", ["cannot", "onboard sensors"]),
            ("Can Spotify tune PID gains?", ["cannot", "controller measurements"]),
            ("Can restaurant ratings replace obstacle sensing?", ["cannot", "geometry"]),
            ("Can a payment app run ROS2 nodes?", ["cannot", "robot compute"]),
            ("Can shopping history replace robot perception?", ["cannot", "sensor data"]),
            ("Can Uber GPS replace SLAM?", ["cannot", "SLAM", "onboard localization"]),
            ("Can Google Maps replace LiDAR?", ["cannot", "range measurements"]),
            ("Can social media replace perception?", ["cannot", "camera or sensor data"]),
            ("Can app permissions replace ROS permissions?", ["cannot", "different systems"]),
            ("Can delivery coupons improve loop closure?", ["cannot", "place-recognition evidence"]),
        ],
        "uncertain": [
            ("Could a proprietary mobility dataset help this robot?", ["depends", "fields", "permissions"]),
            ("Can external app data improve navigation?", ["depends", "data type", "robot task"]),
            ("Would a city platform help localization?", ["uncertain", "measurements", "coordinate frame"]),
            ("Can service logs improve autonomy?", ["depends", "labels", "limitations"]),
            ("Could online records support robot learning?", ["depends", "relevance", "privacy"]),
            ("Can third-party map metadata help this indoor robot?", ["depends", "coverage", "freshness"]),
            ("Would restaurant footfall data help robot scheduling?", ["depends", "task", "time"]),
            ("Can a phone navigation trace help my robot?", ["depends", "coordinate frame", "uncertainty"]),
            ("Could vendor telemetry improve controller tuning?", ["depends", "signals", "permissions"]),
            ("Can an unknown app API support obstacle avoidance?", ["uncertain", "fields", "documentation"]),
        ],
    }
    cases = []
    for relation, items in groups.items():
        expanded = items * 2
        for index, (query, required) in enumerate(expanded, 1):
            cases.append({
                "id": f"cross-{relation}-{index:02d}",
                "suite": "cross_domain_robustness",
                "query": query,
                "gold_relationship": relation,
                "required_points": required,
                "forbidden_claims": [
                    "consumer apps are physical robot sensors",
                    "external data replaces onboard safety or perception without qualification",
                ],
            })
    assert len(cases) == 100
    return cases


def write_benchmarks():
    os.makedirs(BENCHMARK_DIR, exist_ok=True)
    suites = {
        "context_contamination.json": build_context_contamination_suite(),
        "memory_recall.json": build_memory_suite(),
        "knowledge_growth.json": build_knowledge_suite(),
        "intent_classification.json": build_intent_suite(),
        "cross_domain_robustness.json": build_cross_domain_suite(),
    }
    aliases = {
        "context_contamination_suite.json": suites["context_contamination.json"],
        "memory_recall_suite.json": suites["memory_recall.json"],
        "knowledge_growth_suite.json": suites["knowledge_growth.json"],
        "intent_suite.json": suites["intent_classification.json"],
        "cross_domain_robustness_suite.json": suites["cross_domain_robustness.json"],
    }
    suites.update(aliases)
    for filename, rows in suites.items():
        path = os.path.join(BENCHMARK_DIR, filename)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(rows, handle, indent=2, ensure_ascii=False)
    return {name: len(rows) for name, rows in suites.items()}


if __name__ == "__main__":
    for name, count in write_benchmarks().items():
        print(f"{name}: {count}")
