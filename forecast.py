import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import os

def forecast_vitals(csv_path: str, anomaly_percent: float) -> dict:
    """
    Performs real 48-hour time-series forecasting using the ARIMA model on vitals data.
    """
    print(f"Analyzing vitals data: {csv_path} with ARIMA model...")
    
    try:
        df = pd.read_csv(csv_path)
        # Assuming the first column 'T1' is the Heart Rate data (BPM)
        hr_data = df['T1'].astype(float).dropna()
        if len(hr_data) < 10:
            raise ValueError("Insufficient data for forecasting.")
    except Exception as e:
        print(f"Error reading or processing CSV: {e}")
        # Return fallback mock data if real data fails
        return {
            "risk": "Moderate",
            "forecast": "Forecast failed due to data error. Defaulting to Moderate Risk."
        }

    # --- 1. ARIMA Forecasting ---
    
    # Use ARIMA(1, 1, 0) as a simple, stable model (AutoRegressive, Integrated/Differencing, Moving Average)
    order = (1, 1, 0)
    
    try:
        # Fit the model (using the last 100 observations for stability)
        model = ARIMA(hr_data.tail(100), order=order)
        model_fit = model.fit()

        # Generate 48-hour forecast with confidence intervals
        forecast_result = model_fit.get_forecast(steps=48)
        forecast_mean = forecast_result.predicted_mean
        forecast_ci = forecast_result.conf_int() # 95% confidence interval

    except Exception as e:
        print(f"ARIMA model failed to fit: {e}")
        # Fallback if the model fails
        forecast_mean = np.full(48, hr_data.mean())
        forecast_ci = np.array([forecast_mean * 0.95, forecast_mean * 1.05]).T

    # --- 2. Plotting the Forecast ---

    plt.figure(figsize=(10, 5))
    
    # Historical data plot
    history_index = np.arange(len(hr_data))
    plt.plot(history_index, hr_data, label='Historical HR (BPM)', color='blue', alpha=0.7)

    # Forecast index continues from history
    forecast_index = np.arange(len(hr_data), len(hr_data) + 48)
    
    # Forecast plot
    plt.plot(forecast_index, forecast_mean, label='48-Hour Forecast Mean', linestyle='--', color='red')
    
    # Confidence interval plot (filled area)
    plt.fill_between(forecast_index, 
                     forecast_ci.iloc[:, 0].values if isinstance(forecast_ci, pd.DataFrame) else forecast_ci[:, 0], 
                     forecast_ci.iloc[:, 1].values if isinstance(forecast_ci, pd.DataFrame) else forecast_ci[:, 1], 
                     color='red', alpha=0.1, label='95% Confidence Interval')

    plt.title('Heart Rate Forecast: Next 48 Hours (ARIMA)')
    plt.xlabel('Time (Hours)')
    plt.ylabel('Heart Rate (BPM)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.savefig("forecast.png")
    plt.close()

    # --- 3. Determine Risk ---
    
    # Calculate forecast stability (e.g., standard deviation of the forecast mean)
    forecast_volatility = np.std(forecast_mean)
    
    # Risk calculation now depends on both anomaly size AND forecast volatility
    if anomaly_percent >= 2.0 or forecast_volatility > 5.0:
        risk_level = "Critical"
        forecast_summary = f"High probability of instability (Volatility: {forecast_volatility:.2f}). Close monitoring required."
    elif anomaly_percent >= 1.0 or forecast_volatility > 2.0:
        risk_level = "High"
        forecast_summary = f"Significant instability detected (Volatility: {forecast_volatility:.2f}). Prepare contingency plans."
    else:
        risk_level = "Moderate"
        forecast_summary = "Vitals are projected to remain relatively stable. Continue routine monitoring."
        
    return {
        "risk": risk_level,
        "forecast": forecast_summary
    }