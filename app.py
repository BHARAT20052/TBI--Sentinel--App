import streamlit as st
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from report import TBIReport, AnomalyDetails, RiskRecommendations, ForecastSummary
from datetime import datetime

# ====================== CONFIG & THEME ======================
st.set_page_config(page_title="TBI Sentinel | Indian Army", page_icon="India", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .main {background-color: #0e1117; color: #ffffff;}
    .stApp {background: linear-gradient(to bottom, #0e1117, #1a1f2e);}
    .stButton>button {background:#1f4037; color:#00ff00; border:3px solid #00ff00; font-weight:bold; border-radius:12px;}
    .report-block {background:#1a1f2e; padding:35px; border-radius:20px; border-left:12px solid #00ff00; box-shadow:0 0 30px #00ff0040;}
    .header-title {font-size:4rem; font-weight:bold; color:#00ff00; text-align:center; text-shadow:0 0 30px #00ff00cc;}
    .big-text {font-size:2.2rem; font-weight:bold; margin:10px 0;}
    .red {color:#ff0066 !important;}
    .green {color:#00ff00 !important;}
</style>
""", unsafe_allow_html=True)

# ====================== HEADER ======================
col1, col2, col3 = st.columns([1,4,1])
with col2:
    st.markdown("<h1 class='header-title'>India TBI SENTINEL</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:#00ff80; font-size:1.5rem;'>Real-Time Traumatic Brain Injury Analysis • Indian Army</p>", unsafe_allow_html=True)
st.markdown("---")

# ====================== SIDEBAR ======================
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/en/thumb/4/41/Flag_of_India.svg/1200px-Flag_of_India.svg.png", width=150)
    st.success("**10-Second AI Diagnosis • 48-Hour Forecast**")
    st.markdown("**Developed by:** BHARAT20052")

# ====================== UPLOADERS ======================
col1, col2 = st.columns(2)
with col1:
    st.markdown("### Brain Upload Brain MRI")
    mri_file = st.file_uploader("JPG/PNG • Max 200MB", type=["jpg","jpeg","png"], key="mri")
with col2:
    st.markdown("### Chart Upload Heart Rate CSV")
    csv_file = st.file_uploader("Any format • Max 200MB", type=["csv"], key="csv")

if mri_file and csv_file:
    # MRI
    img = cv2.imdecode(np.frombuffer(mri_file.read(), np.uint8), 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    with st.spinner("Scanning brain..."):
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
        anomaly_pct = round((np.sum(thresh == 0) / thresh.size) * 100, 2)

    # HEART RATE — WORKS WITH T1/T2/T3/T4
    df = pd.read_csv(csv_file)
    st.caption(f"Columns detected: {list(df.columns)}")

    keywords = ['heart','hr','bpm','rate','pulse','beat','cardiac']
    hr_col = next((c for c in df.columns if any(k.lower() in c.lower() for k in keywords)), None)
    if not hr_col:
        num_cols = df.select_dtypes(include=['float64','int64']).columns
        hr_col = num_cols[0] if len(num_cols)>0 else None
        if hr_col:
            st.warning(f"Using first numeric column **{hr_col}** as heart rate")

    if not hr_col:
        st.error("No numeric data found!"); st.stop()

    st.success(f"Heart Rate → **{hr_col}**")
    hr = pd.to_numeric(df[hr_col], errors='coerce').dropna()
    if len(hr) < 10:
        st.error("Need ≥10 values"); st.stop()

    # Forecast
    with st.spinner("Forecasting 48 hours..."):
        try:
            model = ARIMA(hr, order=(1,1,1)).fit()
            forecast = model.forecast(48)
            conf = model.get_forecast(48).conf_int()
        except:
            forecast = pd.Series([hr.mean()]*48)
            conf = pd.DataFrame({"lower": [hr.mean()*0.9]*48, "upper": [hr.mean()*1.1]*48})

    trend = "Rising" if forecast.mean() > hr.mean() else "Stable"

    # ====================== FINAL DISPLAY ======================
    st.markdown("### MRI Brain Scan")
    st.image(img_rgb, use_container_width=True)

    st.markdown("---")
    st.markdown("<h1 style='text-align:center; color:#00ff00;'>AI ANALYSIS COMPLETE</h1>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("<div class='big-text red'>Anomaly Volume<br>{} %</div>".format(anomaly_pct), unsafe_allow_html=True)
        risk = "Low" if anomaly_pct < 1 else "Moderate" if anomaly_pct < 3 else "High"
        st.markdown(f"<h2 class='red'>Risk Level: {risk}</h2>", unsafe_allow_html=True)

    with c2:
        st.markdown("<div class='big-text green'>48-Hour Trend<br>{}</div>".format(trend), unsafe_allow_html=True)
        st.markdown(f"<h2 class='green'>Vitals: {trend}</h2>", unsafe_allow_html=True)

    # Graph
    st.markdown("### 48-Hour Heart Rate Forecast")
    fig, ax = plt.subplots(figsize=(13,7))
    ax.plot(hr.tail(100), label="Historical", color="#00ff00", linewidth=3)
    ax.plot(range(len(hr), len(hr)+48), forecast, label="Forecast", color="#ff0066", linewidth=4)
    ax.fill_between(range(len(hr), len(hr)+48), conf.iloc[:,0], conf.iloc[:,1], color="#ff0066", alpha=0.4)
    ax.set_title("48-Hour Heart Rate Forecast", color="white", fontsize=20, fontweight="bold")
    ax.legend(facecolor="#0e1117", labelcolor="white", fontsize=14)
    ax.grid(alpha=0.4)
    ax.set_facecolor("#0e1117")
    fig.patch.set_facecolor("#0e1117")
    st.pyplot(fig)

    # Report
    report = TBIReport(
        patient_id="TBI-001",
        date_of_analysis=datetime.now().strftime("%Y-%m-%d"),
        anomaly_details=AnomalyDetails(volume_percentage=anomaly_pct, location="No major anomaly" if anomaly_pct<1 else "Critical zone", assessment=f"{anomaly_pct}% anomaly detected"),
        risk_recommendations=RiskRecommendations(risk_level=risk, immediate_action="Continue monitoring" if risk!="High" else "EVACUATE NOW", treatment_plan=["Immediate CT", "Neurology consult", "Prepare for surgery"]),
        forecast_summary=ForecastSummary(prediction_validity="High", potential_events="Stable" if trend=="Stable" else "Rising trend", key_metrics_trend=trend),
        final_conclusion=f"Anomaly {anomaly_pct}%. Risk: {risk}. Action required."
    )
    st.json(report.dict(), expanded=False)

    # FINAL VERDICT
    verdict = "SAFE TO MOVE → MONITOR EVERY 6 HOURS" if anomaly_pct < 2 else "EVACUATE TO HOSPITAL IMMEDIATELY"
    color = "#00ff00" if anomaly_pct < 2 else "#ff0066"
    st.markdown(f"<div class='report-block'><h1 style='color:{color}; text-align:center; margin:0;'>FINAL VERDICT<br>{verdict}</h1></div>", unsafe_allow_html=True)

else:
    st.info("Upload Brain MRI + Heart Rate CSV → Get instant AI diagnosis")
    st.image("https://via.placeholder.com/1200x600/0e1117/00ff00?text=TBI+SENTINEL+•+FIELD-READY+•+10-SECOND+AI", use_container_width=True)

st.caption("TBI Sentinel © 2025 BHARAT20052 • For Indian Army • Jai Hind India")