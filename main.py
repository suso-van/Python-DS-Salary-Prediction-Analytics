# main.py
import os
import joblib
from src.pipeline import load_and_prep_data, create_preprocessing_pipeline
from src.train import SalaryModelTrainer
from src.evaluate import ModelEvaluator

if __name__ == "__main__":
    DATA_PATH = os.path.join("data", "salaries.csv")
    TARGET_COLUMN = "salary_in_usd"
    MODEL_DIR = "models"

    print("[Orchestrator] Starting Salary Prediction Analytics Engine...")

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset missing at '{DATA_PATH}'")

    # 1. Map Kaggle schema features
    NUM_FEATURES = ["work_year", "remote_ratio"]
    CAT_FEATURES = ["job_title", "employment_type", "employee_residence", "company_location"]
    ORD_FEATURES = ["experience_level", "company_size"]

    MAPPING_ORDERS = [["EN", "MI", "SE", "EX"], ["S", "M", "L"]]

    # 2. Ingest and split
    (X_train, X_test, y_train, y_test), is_log_transformed = load_and_prep_data(DATA_PATH, TARGET_COLUMN)

    # 3. Build preprocessing pipeline
    preprocessor = create_preprocessing_pipeline(NUM_FEATURES, CAT_FEATURES, ORD_FEATURES, MAPPING_ORDERS)

    # 4. Train architectures
    trainer = SalaryModelTrainer(preprocessor=preprocessor)
    trained_pipelines = trainer.fit_all(X_train, y_train)

    # 5. Evaluate and display metrics
    metrics_summary = ModelEvaluator.generate_metrics_summary(trained_pipelines, X_test, y_test, is_log_transformed)
    
    print("\n========================= MODEL PERFORMANCE METRICS =========================")
    print(metrics_summary)
    print("=============================================================================\n")

    # 6. Automated Model Export Execution
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Identify the model with the highest R² Score
    best_model_name = metrics_summary["R2_Score"].idxmax()
    best_pipeline = trained_pipelines[best_model_name]
    
    model_save_path = os.path.join(MODEL_DIR, "best_salary_pipeline.joblib")
    
    # Save the full end-to-end processing + prediction architecture
    joblib.dump(best_pipeline, model_save_path)
    print(f"[Orchestrator] Success! Exported '{best_model_name}' to '{model_save_path}'.")