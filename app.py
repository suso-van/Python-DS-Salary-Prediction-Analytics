# app.py
import os
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# 1. Initialize FastAPI Application context
app = FastAPI(
    title="Data Science Salary Prediction API",
    description="Production-ready inference endpoint for estimating global market compensation.",
    version="1.0.0"
)

# 2. Define the exact structural input schema using Pydantic
class CandidateProfile(BaseModel):
    work_year: int = Field(default=2026, description="The targeted employment year")
    experience_level: str = Field(default="SE", description="EN (Entry), MI (Mid), SE (Senior), EX (Executive)")
    employment_type: str = Field(default="FT", description="FT, PT, CT, FL")
    job_title: str = Field(default="Data Scientist", description="Raw professional designation")
    employee_residence: str = Field(default="US", description="Two-letter ISO country code of worker")
    remote_ratio: int = Field(default=100, description="0 (On-site), 50 (Hybrid), 100 (Fully Remote)")
    company_location: str = Field(default="US", description="Two-letter ISO country code of employer")
    company_size: str = Field(default="M", description="S (Small), M (Medium), L (Large)")

# 3. Handle model state loading on server startup
MODEL_PATH = os.path.join("models", "best_salary_pipeline.joblib")
if os.path.exists(MODEL_PATH):
    print("[Server] Loading predictive pipeline model state into RAM...")
    model_pipeline = joblib.load(MODEL_PATH)
else:
    model_pipeline = None
    print("[Warning] No serialized pipeline found. Please compile using main.py first.")

@app.get("/")
def health_check():
    """Simple heartbeat endpoint to verify the API gateway status."""
    return {
        "status": "online",
        "model_loaded": model_pipeline is not None
    }

@app.post("/predict")
def predict_salary(profile: CandidateProfile):
    """
    Accepts a standardized JSON candidate payload, runs it through the
    preprocessing transformer layers, and returns the real dollar market value.
    """
    if model_pipeline is None:
        raise HTTPException(
            status_code=503, 
            detail="Prediction pipeline not initialized or file missing on server disk."
        )
    
    try:
        # Convert incoming Pydantic dictionary cleanly into a Pandas DataFrame shape
        input_data = pd.DataFrame([profile.model_dump()])
        
        # Execute prediction pass
        log_prediction = model_pipeline.predict(input_data)[0]
        
        # Revert log scale to standard USD format
        calculated_salary = np.expm1(log_prediction)
        
        return {
            "prediction_status": "success",
            "estimated_salary_usd": round(float(calculated_salary), 2),
            "currency": "USD",
            "per_month_estimate": round(float(calculated_salary / 12), 2)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference Engine Crash: {str(e)}")