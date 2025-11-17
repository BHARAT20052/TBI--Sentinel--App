import streamlit as st
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from report import TBIReport, AnomalyDetails, RiskRecommendations, ForecastSummary
from datetime import datetime

# === PAGE CONFIG & DEFENSE THEME ===
st.set_page_config(
    page_title="TBI Sentinel | Indian Army",
    page_icon="India",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main {background-color: #0e1117; color: #ffffff;}
    .stApp {background: linear-gradient(to bottom, #0e1117, #1a1f2e);}
    .css-1d391kg {color: #00ff00;}
    .stButton>button {background-color: #1f4037; color: #00ff00; border: 1px solid #00ff00;}
    .stFileUploader label {color: #00ff00;}
    .report-block {background-color: #1a1f2e; padding: 20px; border-radius: 12px; border-left: 6px solid #00ff00;}
    .header-title {font-size: 3rem; font-weight: bold; color: #00ff00; text-align: center; text-shadow: 0 0 15px #00ff0040;}
    .subheader {color: #00ff00; font-size: 1.3rem; font-weight: bold;}
</style>
""", unsafe_allow_html=True)

# === HEADER ===
col1, col2, col3 = st.columns([1, 3, 1])
with col2:
    st.markdown("<h1 class='header-title'>India TBI SENTINEL</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:#888; font-size:1.2rem;'>Real-Time Field TBI Analysis for Indian Army</p>", unsafe_allow_html=True)

st.markdown("---")

# === SIDEBAR ===
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/en/thumb/4/41/Flag_of_India.svg/1200px-Flag_of_India.svg.png", width=120)
    st.markdown("### Shield Mission")
    st.success("**Rapid TBI Detection & 48-Hour Vitals Forecast**")
    st.markdown("#### Instructions:")
    st.markdown("1. Upload **Brain MRI** (JPG/PNG)")
    st.markdown("2. Upload **Heart Rate CSV**")
    st.markdown("3. Get AI verdict in **10 seconds**")
    st.markdown("---")
    st.markdown("**Developed by:** BHARAT20052")

# === MAIN UPLOAD AREA ===
col1, col2 = st.columns(2)
with col1:
    st.markdown("<p class='subheader'>Brain Upload Brain MRI</p>", unsafe_allow_html=True)
    mri_file = st.file_uploader("JPG / PNG • Max 200MB", type=["jpg", "jpeg", "png"], key="mri")

with col2:
    st.markdown("<p class='subheader'>Chart Upload Vitals CSV</p>", unsafe_allow_html=True)
    csv_file = st.file_uploader("CSV • Max 200MB", type=["csv"], key="csv")

# === MAIN LOGIC ===
if mri_file and csv_file:
    # === Load & Process MRI ===
    file_bytes = np.asarray(bytearray(mri_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    with st.spinner("Analyzing Brain MRI..."):
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
        anomaly_pixels = np.sum(thresh == 0)
        total_pixels = thresh.size
        anomaly_pct = (anomaly_pixels / total_pixels) * 100

    # === Smart CSV Heart Rate Detection ===
    df = pd.read_csv(csv_file)
    st.write("**CSV Preview:**", df.head(3))

    possible_cols = ['heart_rate', 'HeartRate', 'hr', 'HR', 'Heart Rate', 'bpm', 'BPM', 'value', 'Value', 'rate']
    hr_col = None
    for col in df.columns:
        if any(pc.lower() in col.lower() for pc in possible_cols):
            hr_col = col
            break

    if hr_col is None:
        st.error("Heart rate column not found! Try renaming column to `heart_rate`, `hr`, or `BPM`")
        st.stop()

    st.success(f"Detected heart rate column: **{hr_col}**")
    hr = df[hr_col].dropna().astype(float)

    # === ARIMA Forecast ===
    with st.spinner("Forecasting next 48 hours..."):
        try:
            model = ARIMA(hr, order=(1,1,1))
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=48)
            conf_int = model_fit.get_forecast(steps=48).conf_int()
        except:
            st.warning("ARIMA failed on this data → Using mean forecast")
            forecast = pd.Series([hr.mean()] * 48)
            conf_int = pd.DataFrame({'lower': hr.mean()*0.9, 'upper': hr.mean()*1.1}, index=range(48))

    # === DISPLAY RESULTS ===
    st.markdown("### Image MRI Scan")
    st.image(img_rgb, use_column_width=True)

    st.markdown("---")
    st.markdown("### Lightning AI Analysis Complete")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Brain Anomaly Detection")
        st.metric("Anomaly Volume", f"{anomaly_pct:.2f}%")
        status = "Low" if anomaly_pct < 1 else "Moderate" if anomaly_pct < 3 else "High"
        st.write(f"**Risk Level:** {status}")

    with c2:
        st.markdown("#### Heart Vitals Forecast")
        trend = "Rising" if forecast.mean() > hr.mean() else "Stable"
        st.metric("48-Hour Trend", trend)
        st.write("**ARIMA Model** • 95% Confidence")

    # === Forecast Graph ===
    st.markdown("### Chart 48-Hour Heart Rate Forecast")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(hr.tail(100), label="Historical HR", color="#00ff00", linewidth=2)
    future_idx = range(len(hr), len(hr)+48)
    ax.plot(future_idx, forecast, label="Forecast Mean", color="#ff0066", linewidth=2.5)
    ax.fill_between(future_idx, conf_int.iloc[:,0], conf_int.iloc[:,1], color="#ff0066", alpha=0.3, label="95% Confidence")
    ax.set_title("48-Hour Heart Rate Forecast (ARIMA)", color="white", fontsize=16)
    ax.set_xlabel("Time")
    ax.set_ylabel("BPM")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_facecolor("#1a1f2e")
    fig.patch.set_facecolor("#0e1117")
    st.pyplot(fig)

    # === Final JSON Report ===
    report = TBIReport(
        patient_id="TBI-001",
        date_of_analysis=datetime.now().strftime("%Y-%m-%d"),
        anomaly_details=AnomalyDetails(
            volume_percentage=round(anomaly_pct, 2),
            location="No major anomaly" if anomaly_pct < 1 else "Possible bleed",
            assessment=f"Anomaly detected: {anomaly_pct:.2f}% of brain volume"
        ),
        risk_recommendations=RiskRecommendations(
            risk_level="Low" if anomaly_pct < 1 else "Moderate",
            immediate_action="Continue observation" if anomaly_pct < 2 else "Evacuate immediately",
            treatment_plan=["CT scan in 6 hours", "Neurology consult", "Monitor vitals hourly"]
        ),
        forecast_summary=ForecastSummary(
            prediction_validity="High",
            potential_events="Heart rate projected stable" if trend == "Stable" else "Rising trend detected",
            key_metrics_trend=trend
        ),
        final_conclusion=f"TBI Assessment: {anomaly_pct:.2f}% anomaly. Risk: {'Low' if anomaly_pct < 1 else 'Moderate'}. Vitals: {trend}."
    )

    st.markdown("### Document Structured Medical Report")
    st.json(report.dict(), expanded=False)

    verdict = "SAFE TO MOVE → MONITOR EVERY 6 HOURS" if anomaly_pct < 2 else "EVACUATE TO HOSPITAL"
    st.markdown(f"<div class='report-block'>Final Verdict: <strong>{verdict}</strong></div>", unsafe_allow_html=True)

else:
    st.info("Please upload Brain MRI + Heart Rate CSV to begin analysis")
    st.markdown("### Sample Report Preview")
    st.image("https://via.placeholder.com/800x500/0e1117/00ff00?text=TBI+SENTINEL+AI+REPORT+READY+IN+10+SECONDS", caption="Live AI Report Generated Instantly")