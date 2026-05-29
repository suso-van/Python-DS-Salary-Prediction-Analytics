# tests/test_app.py
import pytest
from fastapi.testclient import TestClient
from app import app

# Initialize the simulated HTTP test client
client = TestClient(app)

def test_api_health_check():
    """Verifies the backend API root gateway boots into an online status."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "online"

def test_api_predict_endpoint_validation_fallback():
    """
    Asserts that sending an invalid data type (e.g., remote_ratio as a string 
    instead of an int) safely returns a 422 Unprocessable Entity status 
    instead of a 500 Server Crash.
    """
    malformed_payload = {
        "work_year": 2026,
        "experience_level": "SE",
        "employment_type": "FT",
        "job_title": "ML Engineer",
        "employee_residence": "US",
        "remote_ratio": "Fully Remote!",  # Structural Error: Expecting Integer
        "company_location": "US",
        "company_size": "M"
    }
    
    response = client.post("/predict", json=malformed_payload)
    assert response.status_code == 422