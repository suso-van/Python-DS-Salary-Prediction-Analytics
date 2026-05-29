# predict.py
import os
import joblib
import pandas as pd
import numpy as np

def live_inference():
    MODEL_PATH = os.path.join("models", "best_salary_pipeline.joblib")
    
    if not os.path.exists(MODEL_PATH):
        print(f"[Error] No trained brain found at {MODEL_PATH}. Run 'python main.py' first.")
        return

    # Load the end-to-end inference pipeline
    print("[Inference] Spawning prediction engine context...")
    pipeline = joblib.load(MODEL_PATH)

    # Mocking a new, incoming candidate profile setup
    # Feel free to change these parameters to test different production outputs
    new_profile = {
        'work_year': [2026],
        'experience_level': ['SE'],          # EN, MI, SE, EX
        'employment_type': ['FT'],           # FT, PT, CT, FL
        'job_title': ['Machine Learning Engineer'], # Cleaned automatically by pipeline if hooked
        'employee_residence': ['US'],        # ISO Country codes
        'remote_ratio': [100],               # 0, 50, 100
        'company_location': ['US'],
        'company_size': ['M']                # S, M, L
    }

    # Convert to DataFrame matching model structural input boundaries
    input_df = pd.DataFrame(new_profile)
    
    # Execute atomic pipeline processing and forward prediction pass
    predicted_log_salary = pipeline.predict(input_df)[0]
    
    # Convert log-space prediction back to actual standard dollar scale
    # (Using expm1 assuming target optimization was active in training)
    final_salary = np.expm1(predicted_log_salary)

    print("\n======================== PREDICTION PROFILE REPORT ========================")
    print(f" Target Position : {new_profile['job_title'][0]} ({new_profile['experience_level'][0]})")
    print(f" Company Profile : Location: {new_profile['company_location'][0]} | Size: {new_profile['company_size'][0]}")
    print(f" Remote Scope    : {new_profile['remote_ratio'][0]}%")
    print("---------------------------------------------------------------------------")
    print(f" Market Evaluated Valuation Estimate: ${final_salary:,.2f} USD / year")
    print("===========================================================================\n")

if __name__ == "__main__":
    live_inference()