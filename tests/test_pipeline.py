import numpy as np
import pandas as pd

from src.pipeline import (
    clean_job_titles,
    create_preprocessing_pipeline,
    load_and_prep_data,
)


def test_clean_job_titles_groups_common_data_roles():
    cases = {
        "Machine Learning Engineer": "ML Engineer",
        "AI Research Scientist": "Data Scientist",
        "Data Warehouse Architect": "Data Engineer",
        "BI Analytics Analyst": "Data Analyst",
        "Director of Data": "Data Leadership",
        "Prompt Wrangler": "Other Tech Specialist",
    }

    for raw_title, expected_group in cases.items():
        assert clean_job_titles(raw_title) == expected_group


def test_load_and_prep_data_drops_leakage_and_log_transforms_skewed_target(tmp_path):
    csv_path = tmp_path / "salaries.csv"
    raw_data = pd.DataFrame(
        {
            "X": [0, 1, 2, 3, 4],
            "work_year": [2023, 2023, 2024, 2024, 2025],
            "experience_level": ["EN", "MI", "SE", "EX", "SE"],
            "employment_type": ["FT", "FT", "CT", "PT", "FT"],
            "job_title": [
                "Data Scientist",
                "Machine Learning Engineer",
                "BI Analyst",
                "Data Engineering Architect",
                "Head of Data",
            ],
            "employee_residence": ["US", "IN", "GB", "CA", "US"],
            "remote_ratio": [100, 50, 0, 100, 50],
            "company_location": ["US", "IN", "GB", "CA", "US"],
            "company_size": ["M", "L", "S", "M", "L"],
            "salary": [100000, 120000, 130000, 140000, 150000],
            "salary_currency": ["USD", "USD", "USD", "USD", "USD"],
            "salary_in_usd": [50000, 60000, 70000, 80000, 1000000],
        }
    )
    raw_data.to_csv(csv_path, index=False)

    (X_train, X_test, y_train, y_test), is_skewed = load_and_prep_data(
        csv_path,
        test_size=0.4,
        random_state=7,
    )

    X_all = pd.concat([X_train, X_test])
    y_all = pd.concat([y_train, y_test])

    assert is_skewed is True
    assert "X" not in X_all.columns
    assert "salary" not in X_all.columns
    assert "salary_currency" not in X_all.columns
    assert set(X_all["job_title"]).issubset(
        {
            "Data Scientist",
            "ML Engineer",
            "Data Analyst",
            "Data Engineer",
            "Data Leadership",
        }
    )
    assert np.allclose(np.sort(y_all.to_numpy()), np.sort(np.log1p(raw_data["salary_in_usd"])))


def test_create_preprocessing_pipeline_outputs_numeric_matrix():
    frame = pd.DataFrame(
        {
            "work_year": [2024, 2025],
            "remote_ratio": [100, 0],
            "job_title": ["Data Scientist", "ML Engineer"],
            "employment_type": ["FT", "CT"],
            "employee_residence": ["US", "IN"],
            "company_location": ["US", "IN"],
            "experience_level": ["SE", "MI"],
            "company_size": ["M", "L"],
        }
    )
    preprocessor = create_preprocessing_pipeline(
        numeric_features=["work_year", "remote_ratio"],
        categorical_features=[
            "job_title",
            "employment_type",
            "employee_residence",
            "company_location",
        ],
        ordinal_features=["experience_level", "company_size"],
        mapping_orders=[["EN", "MI", "SE", "EX"], ["S", "M", "L"]],
    )

    transformed = preprocessor.fit_transform(frame)

    assert transformed.shape[0] == len(frame)
    assert transformed.shape[1] > frame.shape[1]
    assert np.issubdtype(transformed.dtype, np.number)