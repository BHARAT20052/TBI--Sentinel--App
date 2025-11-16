from pydantic import BaseModel
from typing import List

class AnomalyDetails(BaseModel):
    volume_percentage: float
    location: str
    assessment: str

class RiskRecommendations(BaseModel):
    risk_level: str
    immediate_action: str
    treatment_plan: List[str]

class ForecastSummary(BaseModel):
    prediction_validity: str
    potential_events: str
    key_metrics_trend: str

class TBIReport(BaseModel):
    patient_id: str
    date_of_analysis: str
    anomaly_details: AnomalyDetails
    risk_recommendations: RiskRecommendations
    forecast_summary: ForecastSummary
    final_conclusion: str