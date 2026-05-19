import pandas as pd
import math
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

# ================= RISK ZONE =================
RISK_ZONE = {
    "lat_min": 30.7320,
    "lat_max": 30.7400,
    "lon_min": 79.0600,
    "lon_max": 79.0750
}

def is_inside_risk_zone(lat, lon):
    return (
        RISK_ZONE["lat_min"] <= lat <= RISK_ZONE["lat_max"] and
        RISK_ZONE["lon_min"] <= lon <= RISK_ZONE["lon_max"]
    )

# ================= HELPERS =================
def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1-a))

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

    return [avg_speed, total_distance, stagnation_ratio, time_in_zone]

# ================= TRAIN & EVALUATE =================
def train_model():
    df = pd.read_csv("tourist_movement.csv")

    X = df[[
        "avg_speed",
        "total_distance",
        "stagnation_ratio",
        "time_in_zone"
    ]]
    y = df["label"]

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        random_state=42
    )

    model.fit(X_train, y_train)

    # ---------- Evaluation ----------
    y_pred = model.predict(X_test)

    print("\n📊 Classification Report:\n")
    print(classification_report(y_test, y_pred, target_names=encoder.classes_))

    print("\n📉 Confusion Matrix:\n")
    print(confusion_matrix(y_test, y_pred))

    cv_scores = cross_val_score(model, X, y_encoded, cv=5)
    print("\n📈 Cross Validation Accuracy:", cv_scores.mean())

    joblib.dump(model, "risk_model.pkl")
    joblib.dump(encoder, "label_encoder.pkl")

    print("\n✅ Model trained and saved.")

# ================= PREDICT =================
def predict_risk(lat, lon, points):
    if not is_inside_risk_zone(lat, lon):
        return "NORMAL (Outside Risk Zone)"

    model = joblib.load("risk_model.pkl")
    encoder = joblib.load("label_encoder.pkl")

    features = extract_features(points)
    prediction = model.predict([features])

    return encoder.inverse_transform(prediction)[0]

# ================= MAIN =================
if __name__ == "__main__":
    train_model()

    # ---- Manual Test Case ----
    test_points = [
        (30.7350, 79.0650),
        (30.7350, 79.0650),
        (30.7351, 79.0650),
        (30.7351, 79.0650),
        (30.7351, 79.0650),
        (30.7351, 79.0650)
    ]

    lat, lon = test_points[-1]
    print("\n🧪 Predicted Risk Level:",
          predict_risk(lat, lon, test_points))
