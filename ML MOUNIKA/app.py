from flask import Flask, render_template, jsonify, request
import pandas as pd
import joblib
import math

app = Flask(__name__)

# ---------------- Load trained ML model and label encoder ----------------
model = joblib.load("tourist_risk_model.pkl")
encoder = joblib.load("tourist_label_encoder.pkl")

# ---------------- Load GPS data ----------------
gps_data = pd.read_csv("tourist_gps_hourly_data.csv")

# ---------------- Load Risk Zones for Live Demo ----------------
risk_points = pd.read_csv("risk_zones.csv")

# ---------------- Haversine Distance Function ----------------
def haversine(lat1, lon1, lat2, lon2):
    R = 6371000  # Earth radius in meters
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1-a))

# ---------------- Function to check nearest risk zone ----------------
def check_risk(lat, lon):
    min_distance = float("inf")
    current_label = "NORMAL"
    for idx, row in risk_points.iterrows():
        d = haversine(lat, lon, row["latitude"], row["longitude"])
        if d < min_distance:
            min_distance = d
            current_label = row["risk_label"]
    return current_label

# ---------------- Home Route ----------------
@app.route("/")
def home():
    return render_template("index.html")

# ---------------- Hourly Prediction Route ----------------
@app.route("/predict/<int:tourist_id>")
def predict(tourist_id):
    # Filter GPS data for the given tourist
    tourist_data = gps_data[gps_data["tourist_id"] == tourist_id].sort_values("timestamp")

    # Require at least 6 hourly points
    if len(tourist_data) < 6:
        return jsonify({"error": "Not enough GPS data yet"})

    # Calculate total distance and stagnation ratio
    total_distance = 0
    stagnant = 0
    lat_prev = None
    lon_prev = None

    for idx, row in tourist_data.iterrows():
        lat, lon = row["latitude"], row["longitude"]
        if lat_prev is not None:
            d = haversine(lat_prev, lon_prev, lat, lon)
            total_distance += d
            if d < 8:
                stagnant += 1
        lat_prev, lon_prev = lat, lon

    stagnation_ratio = stagnant / (len(tourist_data) - 1)

    # Predict using ML model
    prediction = model.predict([[total_distance, stagnation_ratio]])
    safety_status = encoder.inverse_transform(prediction)[0]

    return jsonify({
        "tourist_id": tourist_id,
        "total_distance_meters": round(total_distance, 2),
        "stagnation_ratio": round(stagnation_ratio, 2),
        "safety_status": safety_status
    })

# ---------------- Live Location Alert Route ----------------
@app.route("/current_location")
def current_location():
    lat = float(request.args.get("lat"))
    lon = float(request.args.get("lon"))

    # Check risk for this location
    risk_status = check_risk(lat, lon)

    return jsonify({
        "latitude": lat,
        "longitude": lon,
        "risk_status": risk_status
    })

# ---------------- Run App ----------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
