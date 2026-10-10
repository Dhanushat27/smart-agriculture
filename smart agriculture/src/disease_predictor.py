# ============================================================
# SUPERVISED ML #2 - Disease Risk Prediction
# Data generated synthetically from real agronomic rules
# ============================================================
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

os.makedirs("models", exist_ok=True)
np.random.seed(42)

# ---------- Generate Synthetic Sensor Data ----------
n = 6000
temp = np.random.uniform(10, 45, n)           # Temperature (°C)
humidity = np.random.uniform(20, 100, n)      # Humidity (%)
soil_moisture = np.random.uniform(10, 90, n)  # Soil Moisture (%)
ph = np.random.uniform(4, 9, n)               # Soil pH

def label(t, h, sm, p):
    if h > 85 and 15 <= t <= 25:  return "Late Blight Risk"     # cool + very humid
    if h > 70 and 25 < t <= 32:   return "Early Blight Risk"    # warm + humid
    if h < 45 and t > 28:         return "Powdery Mildew Risk"  # hot + dry
    if sm > 80:                   return "Root Rot Risk"        # waterlogged soil
    if p < 5 or p > 8:            return "Nutrient Lockout Risk"
    return "Healthy"

y = [label(t, h, s, p) for t, h, s, p in zip(temp, humidity, soil_moisture, ph)]

FEATURES = ["temp", "humidity", "soil_moisture", "ph"]
X = pd.DataFrame({"temp": temp, "humidity": humidity,
                  "soil_moisture": soil_moisture, "ph": ph})

print("Class distribution:\n", pd.Series(y).value_counts())

# ---------- Train / Test ----------
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_tr, y_tr)

pred = model.predict(X_te)
print("\nAccuracy:", accuracy_score(y_te, pred))
print(classification_report(y_te, pred))

# ---------- Save ----------
joblib.dump(model, "models/disease_model.joblib")
print("✅ Saved -> models/disease_model.joblib")