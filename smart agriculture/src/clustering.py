# ============================================================
# UNSUPERVISED ML - K-Means Clustering (Elbow Method)
# Groups plants/zones by soil & climate conditions
# ============================================================
import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

os.makedirs("models", exist_ok=True)

# ---------- Load + Scale ----------
df = pd.read_csv("data/Crop_recommendation.csv")
FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
X = df[FEATURES]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------- Elbow Method (find best k) ----------
inertia = []
K_RANGE = range(1, 11)
for k in K_RANGE:
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    km.fit(X_scaled)
    inertia.append(km.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(K_RANGE, inertia, "bo-")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia (WCSS)")
plt.title("Elbow Method for Optimal k")
plt.grid(alpha=0.3)
plt.savefig("elbow.png", dpi=120, bbox_inches="tight")
print("✅ Saved -> elbow.png")

# ---------- Final Model (k = 4) ----------
kmeans = KMeans(n_clusters=4, n_init=10, random_state=42)
df["cluster"] = kmeans.fit_predict(X_scaled)

print("\nCluster centers (interpret each zone):")
print(pd.DataFrame(kmeans.cluster_centers_, columns=FEATURES))

print("\nSamples per cluster:")
print(df["cluster"].value_counts())

# ---------- Save ----------
joblib.dump(kmeans, "models/kmeans_model.joblib")
joblib.dump(scaler, "models/kmeans_scaler.joblib")
print("✅ Saved -> models/kmeans_model.joblib")