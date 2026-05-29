# predict.py
import os
import joblib
import pandas as pd
import numpy as np

def get_terminal_input(prompt, options=None, default=None):
    """Helper to capture clean user input with default fail-safes."""
    if options:
        prompt_text = f"{prompt} ({'/'.join(options)}) [{default}]: "
    else:
        prompt_text = f"{prompt} [{default}]: "
        
    user_input = input(prompt_text).strip().upper()
    
    if not user_input and default:
        return default
    if options and user_input not in options:
        print(f" Invalid selection. Defaulting to '{default}'")
        return default
    return user_input

def live_interactive_inference():
    MODEL_PATH = os.path.join("models", "best_salary_pipeline.joblib")
    
    if not os.path.exists(MODEL_PATH):
        print(f"[Error] No trained brain found at {MODEL_PATH}. Run 'python main.py' first.")
        return

    print("\n==========================================================================")
    print("      INITIALIZING SALARY PREDICTION ANALYTICS TERMINAL INTERFACE         ")
    print("==========================================================================\n")
    
    pipeline = joblib.load(MODEL_PATH)

    # Gather inputs dynamically from the user
    work_year = input("Target Job Year [2026]: ").strip()
    work_year = int(work_year) if work_year.isdigit() else 2026

    experience_level = get_terminal_input("Experience Level", ["EN", "MI", "SE", "EX"], "SE")
    employment_type = get_terminal_input("Employment Type", ["FT", "PT", "CT", "FL"], "FT")
    
    job_title = input("Enter Raw Job Title [Data Scientist]: ").strip()
    if not job_title:
        job_title = "Data Scientist"
        
    employee_residence = input("Candidate Residence Country Code (ISO) [US]: ").strip().upper() or "US"
    
    remote_ratio = input("Remote Percentage (0, 50, 100) [100]: ").strip()
    remote_ratio = int(remote_ratio) if remote_ratio in ["0", "50", "100"] else 100
    
    company_location = input("Company Location Country Code (ISO) [US]: ").strip().upper() or "US"
    company_size = get_terminal_input("Company Size", ["S", "M", "L"], "M")

    # Pack input array structure
    profile = {
        'work_year': [work_year],
        'experience_level': [experience_level],
        'employment_type': [employment_type],
        'job_title': [job_title],
        'employee_residence': [employee_residence],
        'remote_ratio': [remote_ratio],
        'company_location': [company_location],
        'company_size': [company_size]
    }

    input_df = pd.DataFrame(profile)
    
    # Forward model prediction pass
    predicted_log_salary = pipeline.predict(input_df)[0]
    final_salary = np.expm1(predicted_log_salary)

    print("\n======================== LIVE INFERENCE REPORT ========================")
    print(f" Target Position : {job_title} ({experience_level})")
    print(f" Scope           : {remote_ratio}% Remote | Type: {employment_type}")
    print(f" Geography       : Candidate: {employee_residence} -> Company: {company_location} ({company_size})")
    print("-----------------------------------------------------------------------")
    print(f" Market Valuation Estimate: ${final_salary:,.2f} USD / year")
    print("=======================================================================\n")

if __name__ == "__main__":
    live_interactive_inference()