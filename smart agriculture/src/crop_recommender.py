# ============================================================
# SUPERVISED ML - Crop Recommendation (Classification)
# Dataset: Crop_recommendation.csv
# ============================================================
import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, f1_score,
                             classification_report, confusion_matrix)
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

os.makedirs("models", exist_ok=True)

# ---------- Load Data ----------
df = pd.read_csv("data/Crop_recommendation.csv")
print("Dataset shape:", df.shape)

FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
X, y = df[FEATURES], df["label"]

# ---------- Train/Test Split (80/20, stratified) ----------
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

# ---------- Compare 3 Models ----------
models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "SVM": SVC(),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
}

results = {}
for name, m in models.items():
    m.fit(X_tr, y_tr)
    pred = m.predict(X_te)
    acc = accuracy_score(y_te, pred)
    f1 = f1_score(y_te, pred, average="macro")
    results[name] = (acc, f1)
    print(f"{name:15}  Accuracy = {acc:.4f}   F1(macro) = {f1:.4f}")

# ---------- Detailed Report of Best Model ----------
best_name = max(results, key=lambda k: results[k][0])
best = models[best_name]
print(f"\nBest Model: {best_name}\n")
print(classification_report(y_te, best.predict(X_te)))
print("Confusion Matrix:")
print(confusion_matrix(y_te, best.predict(X_te)))

# ---------- Save Model ----------
joblib.dump(best, "models/crop_model.joblib")
print("\n✅ Saved -> models/crop_model.joblib")