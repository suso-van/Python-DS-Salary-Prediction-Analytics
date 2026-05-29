# app.py
import os
import sqlite3
import datetime
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Security, Depends, BackgroundTasks
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

API_KEY_NAME = "X-API-KEY"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)
SYSTEM_SECRET_TOKEN = os.getenv("API_ACCESS_TOKEN", "default_fallback_secret")

app = FastAPI(
    title="Data Science Salary Prediction API (Ultra-Advanced)",
    description="Asynchronous event-driven inference gateway with dynamic hot-reloading capability.",
    version="2.0.0"
)

DB_PATH = "inference_audit.db"
MODEL_PATH = os.path.join("models", "best_salary_pipeline.joblib")

# Global volatile memory container for the live model state
LIVE_SYSTEM_STATE = {
    "model_pipeline": None,
    "last_loaded": None
}

class CandidateProfile(BaseModel):
    work_year: int = Field(default=2026)
    experience_level: str = Field(default="SE")
    employment_type: str = Field(default="FT")
    job_title: str = Field(default="Data Scientist")
    employee_residence: str = Field(default="US")
    remote_ratio: int = Field(default=100)
    company_location: str = Field(default="US")
    company_size: str = Field(default="M")

def load_model_into_ram():
    """Safely pulls the binary model matrix into memory without downing the engine."""
    if os.path.exists(MODEL_PATH):
        LIVE_SYSTEM_STATE["model_pipeline"] = joblib.load(MODEL_PATH)
        LIVE_SYSTEM_STATE["last_loaded"] = datetime.datetime.now().isoformat()
        print(f"[Core] Hot-Reload Successful. Model active state updated: {LIVE_SYSTEM_STATE['last_loaded']}")
        return True
    return False

# Initial boot-up load
load_model_into_ram()

def get_api_key(api_key: str = Depends(api_key_header)):
    if api_key == SYSTEM_SECRET_TOKEN:
        return api_key
    raise HTTPException(status_code=403, detail="Access Denied: Invalid Header Token.")

def async_db_logger(profile_dict: dict, calculated_salary: float):
    """
    NON-BLOCKING WORKER THREAD:
    Executes relational database commits outside the main client request loop.
    """
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO inference_logs (
                    timestamp, work_year, experience_level, employment_type, 
                    job_title, employee_residence, remote_ratio, company_location, 
                    company_size, predicted_salary_usd
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.datetime.now().isoformat(),
                profile_dict['work_year'],
                profile_dict['experience_level'],
                profile_dict['employment_type'],
                profile_dict['job_title'],
                profile_dict['employee_residence'],
                profile_dict['remote_ratio'],
                profile_dict['company_location'],
                profile_dict['company_size'],
                calculated_salary
            ))
            conn.commit()
    except Exception as e:
        # Prevents database errors from crashing active user inference requests
        print(f"[Asynchronous Log Error] Thread write skipped: {str(e)}")

@app.get("/")
def health_check():
    return {
        "status": "online",
        "engine_version": "2.0.0",
        "model_timestamp": LIVE_SYSTEM_STATE["last_loaded"]
    }

@app.post("/predict", dependencies=[Depends(get_api_key)])
def predict_salary(profile: CandidateProfile, background_tasks: BackgroundTasks):
    """
    High-Throughput Inference Route:
    Computes mathematical valuation and offloads database logging to a background thread.
    """
    if LIVE_SYSTEM_STATE["model_pipeline"] == None:
        raise HTTPException(status_code=503, detail="Inference pipeline uninitialized.")
    
    try:
        profile_dict = profile.model_dump()
        input_data = pd.DataFrame([profile_dict])
        
        # Immediate evaluation pass
        log_prediction = LIVE_SYSTEM_STATE["model_pipeline"].predict(input_data)[0]
        calculated_salary = round(float(np.expm1(log_prediction)), 2)
        
        # Offload Disk I/O out of the main thread execution line
        background_tasks.add_task(async_db_logger, profile_dict, calculated_salary)
        
        return {
            "prediction_status": "success",
            "estimated_salary_usd": calculated_salary,
            "latency_optimization": "asynchronous_logging_active"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference Crash: {str(e)}")

@app.post("/model/reload", dependencies=[Depends(get_api_key)])
def reload_model_gateway():
    """
    Zero-Downtime Administration Route:
    Swaps out the in-RAM model file seamlessly while the server remains hot.
    """
    success = load_model_into_ram()
    if success:
        return {
            "status": "synchronized",
            "message": "New model pipeline injected successfully without server disruption.",
            "active_version_timestamp": LIVE_SYSTEM_STATE["last_loaded"]
        }
    raise HTTPException(status_code=404, detail="Target pipeline artifact file missing on disk.")