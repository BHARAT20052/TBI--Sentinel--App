from pydantic import BaseModel, Field
from typing import List

class AnomalyDetails(BaseModel):
    """Details about the brain anomaly detected."""
    volume_percentage: float = Field(description="The estimated volume percentage of the anomaly (e.g., hemorrhage, edema).")
    location: str = Field(description="The primary location of the anomaly (e.g., 'Right Frontal Lobe', 'Brain Stem').")
    assessment: str = Field(description="A brief, initial medical assessment of the anomaly's severity and potential impact.")

class RiskRecommendations(BaseModel):
    """Recommendations based on the calculated patient risk."""
    risk_level: str = Field(description="The categorical risk level (e.g., 'Low', 'Moderate', 'High', 'Critical').")
    immediate_action: str = Field(description="The most critical immediate medical action required (e.g., 'Monitor ICP closely', 'Prepare for surgical intervention').")
    treatment_plan: List[str] = Field(description="A list of 3-5 recommended follow-up diagnostic or treatment steps.")

class ForecastSummary(BaseModel):
    """Summary of the 48-hour vital sign forecast."""
    prediction_validity: str = Field(description="A statement on the confidence/validity of the forecast (e.g., 'High confidence due to stable baseline', 'Low confidence due to high variability').")
    potential_events: str = Field(description="Predicted critical events within 48 hours based on the forecast trend (e.g., 'No critical events projected', 'High probability of significant heart rate drop within 12-24 hours').")
    key_metrics_trend: str = Field(description="A summary of the trend in key metrics (e.g., 'Heart rate shows a slight upward trend', 'Blood pressure remains stable').")

class TBIReport(BaseModel):
    """The final structured clinical report for a Traumatic Brain Injury (TBI) patient."""
    patient_id: str = Field(default="TBI-001", description="A unique identifier for the patient case.")
    date_of_analysis: str = Field(default="YYYY-MM-DD", description="The date the analysis was performed.")
    anomaly_details: AnomalyDetails
    risk_recommendations: RiskRecommendations
    forecast_summary: ForecastSummary
    final_conclusion: str = Field(description="A single, synthesizing paragraph for the final clinical summary and triage recommendation.")