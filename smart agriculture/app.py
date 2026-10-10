# ============================================================
# STREAMLIT DASHBOARD - Smart Agriculture & Crop Management
# Run:  streamlit run app.py
# ============================================================
import os
import numpy as np
import pandas as pd
import streamlit as st
import joblib

st.set_page_config(page_title="Smart Agriculture", page_icon="🌱")
st.title("🌱 Smart Agriculture & Crop Management System")
st.caption("Machine Learning Project — Supervised + Unsupervised + Reinforcement Learning")

# ---------- Check models exist ----------
REQUIRED = ["models/crop_model.joblib", "models/disease_model.joblib",
            "models/kmeans_model.joblib", "models/irrigation_q.npy"]
missing = [f for f in REQUIRED if not os.path.exists(f)]
if missing:
    st.error("⚠️ Missing files: " + ", ".join(missing) +
             "\n\nRun the 4 training scripts in /src first!")
    st.stop()

tab1, tab2, tab3, tab4 = st.tabs(
    ["🌾 Crop Recommendation", "🦠 Disease Risk", "🔍 Clustering", "💧 Smart Irrigation"])

# ================= TAB 1: CROP RECOMMENDATION =================
with tab1:
    st.subheader("🌾 Crop Recommendation (Supervised ML — Random Forest)")
    FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

    c1, c2, c3 = st.columns(3)
    N  = c1.slider("Nitrogen (N)", 0, 140, 50)
    P  = c2.slider("Phosphorus (P)", 5, 145, 40)
    K  = c3.slider("Potassium (K)", 5, 205, 40)
    c1, c2, c3 = st.columns(3)
    temp = c1.slider("Temperature (°C)", 0.0, 50.0, 25.0)
    hum  = c2.slider("Humidity (%)", 10.0, 100.0, 60.0)
    ph   = c3.slider("Soil pH", 3.5, 10.0, 6.5)
    rain = st.slider("Rainfall (mm)", 0.0, 300.0, 100.0)

    if st.button("🌾 Recommend Crop"):
        model = joblib.load("models/crop_model.joblib")
        input_df = pd.DataFrame([[N, P, K, temp, hum, ph, rain]], columns=FEATURES)
        pred = model.predict(input_df)[0]
        proba = model.predict_proba(input_df).max()
        st.success(f"### ✅ Recommended Crop: **{pred.upper()}**")
        st.info(f"Model confidence: **{proba:.1%}**")

# ================= TAB 2: DISEASE RISK =================
with tab2:
    st.subheader("🦠 Disease Risk Prediction (Supervised ML)")

    c1, c2 = st.columns(2)
    t  = c1.slider("Temperature (°C)", 10.0, 45.0, 25.0)
    h  = c2.slider("Humidity (%)", 20.0, 100.0, 60.0)
    c1, c2 = st.columns(2)
    sm = c1.slider("Soil Moisture (%)", 10.0, 90.0, 40.0)
    p  = c2.slider("Soil pH", 4.0, 9.0, 6.5)

    if st.button("🦠 Predict Disease Risk"):
        model = joblib.load("models/disease_model.joblib")
        input_df = pd.DataFrame([[t, h, sm, p]],
                                columns=["temp", "humidity", "soil_moisture", "ph"])
        pred = model.predict(input_df)[0]
        if pred == "Healthy":
            st.success(f"### ✅ Status: **{pred}** — conditions are safe")
        else:
            st.error(f"### ⚠️ Risk Detected: **{pred}**")

# ================= TAB 3: CLUSTERING =================
with tab3:
    st.subheader("🔍 Unsupervised Learning — K-Means Clustering")
    st.write("Plants/zones grouped into **4 clusters** based on soil & climate "
             "conditions → each cluster needs different care (water/fertilizer).")

    if os.path.exists("elbow.png"):
        st.image("elbow.png", caption="Elbow Method — used to choose k")
    else:
        st.warning("Run src/clustering.py first to generate elbow.png")

    if st.checkbox("Show cluster centers"):
        model = joblib.load("models/kmeans_model.joblib")
        feats = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
        st.dataframe(pd.DataFrame(model.cluster_centers_, columns=feats).round(2))

# ================= TAB 4: RL IRRIGATION =================
with tab4:
    st.subheader("💧 Smart Irrigation Agent (Reinforcement Learning — Q-Learning)")
    st.write("The agent learned **when and how much to water** so soil moisture "
             "stays near the optimal level (6).")

    Q = np.load("models/irrigation_q.npy")
    m = st.slider("Current Soil Moisture (0 = dry, 10 = soaked)", 0, 10, 3)

    action = int(np.argmax(Q[m]))
    st.metric("Agent Recommends:", f"Water {action} unit(s)")

    if os.path.exists("rl_reward.png"):
        st.image("rl_reward.png", caption="Reward curve — agent improves with training")

st.divider()
st.caption("Smart Agriculture & Crop Management System | ML Project")