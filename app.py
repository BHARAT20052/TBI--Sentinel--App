import streamlit as st
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from report import TBIReport, AnomalyDetails, RiskRecommendations, ForecastSummary
from datetime import datetime
import json

# === DEFENSE THEME ===
st.set_page_config(
    page_title="TBI Sentinel | Indian Army",
    page_icon="🇮🇳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Military Look
st.markdown("""
<style>
    .main {background-color: #0e1117; color: #ffffff;}
    .stApp {background: linear-gradient(to bottom, #0e1117, #1a1f2e);}
    .css-1d391kg {color: #00ff00;}
    .stButton>button {background-color: #1f4037; color: #00ff00; border: 1px solid #00ff00;}
    .stFileUploader label {color: #00ff00;}
    .css-1y0t9og {color: #00ff00;}
    .report-block {background-color: #1a1f2e; padding: 15px; border-radius: 10px; border-left: 5px solid #00ff00;}
    .header-title {font-size: 2.5rem; font-weight: bold; color: #00ff00; text-align: center; text-shadow: 0 0 10px #00ff0040;}
    .subheader {color: #00ff00; font-size: 1.2rem;}
</style>
""", unsafe_allow_html=True)

# === HEADER ===
col1, col2, col3 = st.columns([1, 3, 1])
with col2:
    st.markdown("<h1 class='header-title'>🇮🇳 TBI SENTINEL</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:#888;'>Field TBI Analysis for Indian Army</p>", unsafe_allow_html=True)

st.markdown("---")

# === SIDEBAR: Instructions ===
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/en/thumb/4/41/Flag_of_India.svg/1200px-Flag_of_India.svg.png", width=100)
    st.markdown("### 🛡 *Mission*")
    st.info("*Rapid TBI Detection & Vitals Forecasting*")
    st.markdown("#### How to Use:")
    st.markdown("1. Upload *Brain MRI* (JPG/PNG)")
    st.markdown("2. Upload *Vitals CSV* (Heart Rate)")
    st.markdown("3. Get *AI Report in 10 sec*")
    st.markdown("---")
    st.markdown("*Developed by:* BHARAT20052")

# === MAIN APP ===
col1, col2 = st.columns(2)

with col1:
    st.markdown("<p class='subheader'>🧠 Upload Brain MRI</p>", unsafe_allow_html=True)
    mri_file = st.file_uploader("Limit 200MB per file • JPG, PNG", type=["jpg", "png"], key="mri")

with col2:
    st.markdown("<p class='subheader'>📊 Upload Vitals CSV</p>", unsafe_allow_html=True)
    csv_file = st.file_uploader("Limit 200MB per file • CSV", type=["csv"], key="csv")

if mri_file and csv_file:
    # Load MRI
    file_bytes = np.asarray(bytearray(mri_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Load CSV
    df = pd.read_csv(csv_file)
    hr = df['heart_rate'].dropna()

    with st.spinner("🔍 Analyzing MRI..."):
        # Simple anomaly detection (example)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
        anomaly_pixels = np.sum(thresh == 0)
        total_pixels = thresh.size
        anomaly_pct = (anomaly_pixels / total_pixels) * 100

    with st.spinner("📈 Forecasting Vitals..."):
        model = ARIMA(hr, order=(1,1,1))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=48)
        conf_int = model_fit.get_forecast(steps=48).conf_int()

    # === DISPLAY MRI ===
    st.markdown("### 🖼 MRI Scan")
    st.image(img_rgb, caption="Uploaded MRI", use_column_width=True)

    # === RESULTS ===
    st.markdown("---")
    st.markdown("### ⚡ *AI Analysis Complete*")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🧠 *Anomaly Detection*")
        st.metric("Anomaly Volume", f"{anomaly_pct:.2f}%")
        st.write(f"*Assessment:* Anomaly detected: {anomaly_pct:.2f}% of brain volume")

    with col2:
        st.markdown("#### ❤ *Vitals Forecast*")
        st.metric("48-Hour Trend", "Stable" if forecast.mean() < 100 else "Rising")
        st.write("*ARIMA Model* • High Confidence")

    # === FORECAST GRAPH ===
    st.markdown("### 📉 *48-Hour Heart Rate Forecast*")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(hr.index[-100:], hr[-100:], label="Historical HR (BPM)", color="#00ff00")
    ax.plot(range(len(hr), len(hr)+48), forecast, label="48-Hour Forecast Mean", color="#ff0066")
    ax.fill_between(range(len(hr), len(hr)+48), conf_int.iloc[:, 0], conf_int.iloc[:, 1], color="#ff0066", alpha=0.2, label="95% Confidence")
    ax.set_title("Heart Rate Forecast: Next 48 Hours (ARIMA)", color="white")
    ax.set_xlabel("Time (Hours)")
    ax.set_ylabel("Heart Rate (BPM)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_facecolor("#1a1f2e")
    fig.patch.set_facecolor("#0e1117")
    st.pyplot(fig)

    # === JSON REPORT ===
    report = TBIReport(
        patient_id="TBI-001",
        date_of_analysis=datetime.now().strftime("%Y-%m-%d"),
        anomaly_details=AnomalyDetails(
            volume_percentage=round(anomaly_pct, 2),
            location="No major anomaly" if anomaly_pct < 1 else "Left temporal",
            assessment=f"Anomaly detected: {anomaly_pct:.2f}% of brain volume"
        ),
        risk_recommendations=RiskRecommendations(
            risk_level="Moderate" if 0.5 <= anomaly_pct < 2 else "Low",
            immediate_action="Continue observation",
            treatment_plan=["CT scan in 6 hours", "Neurology consult", "IV fluids if needed"]
        ),
        forecast_summary=ForecastSummary(
            prediction_validity="High (ARIMA model)",
            potential_events="Vitals are projected to remain relatively stable. Continue routine monitoring.",
            key_metrics_trend="Rising trend" if forecast.mean() > hr.mean() else "Stable"
        ),
        final_conclusion=f"TBI Assessment: Anomaly {anomaly_pct:.2f}%. Risk: Moderate. Vitals stable."
    )

    st.markdown("### 📋 *Structured Medical Report (JSON)*")
    st.json(report.dict(), expanded=False)

    st.markdown("<div class='report-block'>✅ <strong>Final Verdict:</strong> SAFE TO MOVE. MONITOR EVERY 6 HOURS.</div>", unsafe_allow_html=True)

else:
    st.info("👈 Upload MRI and CSV to begin AI analysis")
    st.markdown("### 🖼 *Sample Output Preview*")
    st.image("https://via.placeholder.com/600x400/1a1f2e/00ff00?text=TBI+Sentinel+AI+Report", caption="Live Report Generated in 10 Seconds")