import random
import math
import csv

# ================= RISK ZONE =================
RISK_ZONE = {
    "lat_min": 30.7320,
    "lat_max": 30.7400,
    "lon_min": 79.0600,
    "lon_max": 79.0750
}

# ================= HELPERS =================
def haversine(lat1, lon1, lat2, lon2):
    R = 6371000  # meters
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1-a))

def simulate_session():
    points = []
    lat = random.uniform(RISK_ZONE["lat_min"], RISK_ZONE["lat_max"])
    lon = random.uniform(RISK_ZONE["lon_min"], RISK_ZONE["lon_max"])

    for _ in range(6):  # 1 hour (every 10 mins)
        mode = random.choice(["normal", "slow", "stagnant"])

        if mode == "normal":
            lat += random.uniform(0.0004, 0.0008)
            lon += random.uniform(0.0004, 0.0008)
        elif mode == "slow":
            lat += random.uniform(0.00005, 0.00015)
            lon += random.uniform(0.00005, 0.00015)
        else:
            lat += random.uniform(-0.00001, 0.00001)
            lon += random.uniform(-0.00001, 0.00001)

        points.append((lat, lon))

    return points

def extract_features(points):
    total_distance = 0
    stagnant = 0

    for i in range(1, len(points)):
        d = haversine(
            points[i-1][0], points[i-1][1],
            points[i][0], points[i][1]
        )
        total_distance += d
        if d < 10:
            stagnant += 1

    avg_speed = total_distance / 3600
    stagnation_ratio = stagnant / (len(points) - 1)
    time_in_zone = 60

    return avg_speed, total_distance, stagnation_ratio, time_in_zone

def assign_label(stagnation_ratio, time_in_zone):
    if time_in_zone >= 60 and stagnation_ratio > 0.6:
        return "HIGH_RISK"
    elif time_in_zone >= 30:
        return "WARNING"
    else:
        return "NORMAL"

# ================= CREATE DATASET =================
with open("tourist_movement.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "avg_speed",
        "total_distance",
        "stagnation_ratio",
        "time_in_zone",
        "label"
    ])

    for _ in range(1000):
        session = simulate_session()
        avg_speed, dist, stag, time = extract_features(session)
        label = assign_label(stag, time)
        writer.writerow([avg_speed, dist, stag, time, label])

print("✅ Dataset generated: tourist_movement.csv")
