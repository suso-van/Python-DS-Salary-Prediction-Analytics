# tests/test_app.py
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)
HEADERS = {"X-API-KEY": "ghost_architect_secure_token_2026"}

def test_api_health_check_status():
    """Asserts that the public health check endpoint responds successfully."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "online"

def test_api_predict_endpoint_validation_fallback():
    """
    Asserts that sending an invalid data type safely returns a 422 Unprocessable Entity
    once cleared by the security layer.
    """
    malformed_payload = {
        "work_year": 2026,
        "experience_level": "SE",
        "employment_type": "FT",
        "job_title": "ML Engineer",
        "employee_residence": "US",
        "remote_ratio": "Fully Remote!",  # Structural Error: Should be Integer
        "company_location": "US",
        "company_size": "M"
    }
    
    # Passing valid security token but bad data structure
    response = client.post("/predict", json=malformed_payload, headers=HEADERS)
    assert response.status_code == 422