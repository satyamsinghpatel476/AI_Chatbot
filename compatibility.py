# compatibility.py

def compatibility_check(query):

    q = query.lower()

    robotics_keywords = [
        "slam",
        "pid",
        "robot",
        "robotics",
        "localization",
        "navigation",
        "navigation stack",
        "path planning",
        "ros",
        "ros2",
        "sensor",
        "sensors",
        "lidar",
        "kalman",
        "ekf",
        "amcl",
        "gazebo",
        "wheel encoder",
        "encoder",
        "odometry",
        "imu",
        "controller",
        "control",
        "mapping",
        "state estimation"
    ]

    daily_keywords = [
        "uber",
        "ola",
        "lyft",
        "ride",
        "cab",
        "taxi",
        "food",
        "zomato",
        "swiggy",
        "google maps",
        "maps",
        "delivery",
        "whatsapp",
        "instagram",
        "spotify",
        "phonepe",
        "google pay",
        "blinkit",
        "zepto",
        "music",
        "payment",
        "groceries"
    ]

    robotics_found = any(
        k in q for k in robotics_keywords
    )

    daily_found = any(
        k in q for k in daily_keywords
    )

    if robotics_found and daily_found:
        return False

    return True
