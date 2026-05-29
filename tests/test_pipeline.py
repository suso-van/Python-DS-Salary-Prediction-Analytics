# tests/test_pipeline.py
import pytest
from src.pipeline import clean_job_titles

def test_clean_job_titles_ml():
    """Asserts that various machine learning strings map cleanly to 'ML Engineer'"""
    assert clean_job_titles("Lead Machine Learning Engineer") == "ML Engineer"
    assert clean_job_titles("ml specialist") == "ML Engineer"
    assert clean_job_titles("Computer Vision Researcher") == "ML Engineer"

def test_clean_job_titles_ds():
    """Asserts that data science variations map cleanly to 'Data Scientist'"""
    assert clean_job_titles("Data Scientist II") == "Data Scientist"
    assert clean_job_titles("AI Research Scientist") == "Data Scientist"

def test_clean_job_titles_fallback():
    """Asserts that unmapped edge-case roles fallback safely to the tech specialist bucket"""
    assert clean_job_titles("Quantum Computing Dev") == "Other Tech Specialist"