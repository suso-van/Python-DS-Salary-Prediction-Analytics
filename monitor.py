# monitor.py
import os
import sqlite3
import pandas as pd
import numpy as np

DB_PATH = "inference_audit.db"
BASELINE_PATH = os.path.join("data", "salaries.csv")  # Assuming your original raw data is here
DRIFT_THRESHOLD = 0.15  # 15% Max allowable distribution shift before warning

def calculate_categorical_drift(baseline_df, production_df, column):
    """Calculates the proportional distribution difference for a categorical column."""
    if column not in baseline_df.columns or column not in production_df.columns:
        return None
    
    # Calculate relative frequencies (percentages of the total dataset)
    base_dist = baseline_df[column].value_counts(normalize=True).to_dict()
    prod_dist = production_df[column].value_counts(normalize=True).to_dict()
    
    all_keys = set(base_dist.keys()).union(set(prod_dist.keys()))
    
    max_shift = 0.0
    for key in all_keys:
        base_val = base_dist.get(key, 0.0)
        prod_val = prod_dist.get(key, 0.0)
        shift = abs(base_val - prod_val)
        if shift > max_shift:
            max_shift = shift
            
    return max_shift

def run_health_monitor():
    print("\n==========================================================================")
    print("                MLOPS AUTOMATED ECOSYSTEM HEALTH MONITOR                  ")
    print("==========================================================================\n")
    
    # 1. Ensure both data sources are present
    if not os.path.exists(BASELINE_PATH):
        print(f"[Error] Baseline dataset missing at {BASELINE_PATH}. Cannot compute drift.")
        return
    if not os.path.exists(DB_PATH):
        print(f"[Error] Production audit database missing at {DB_PATH}. Run some API traffic first.")
        return

    # 2. Extract Dataframes
    base_df = pd.read_csv(BASELINE_PATH)
    
    with sqlite3.connect(DB_PATH) as conn:
        prod_df = pd.read_sql_query("SELECT * FROM inference_logs", conn)
        
    if len(prod_df) < 10:
        print(f"[Info] Production sample size too small ({len(prod_df)} records). Wait for more traffic.")
        return

    print(f" Loaded {len(base_df)} baseline training instances.")
    print(f" Loaded {len(prod_df)} production transaction logs.")
    print("--------------------------------------------------------------------------")
    print(f" Checking system stability (Max Drift Tolerance: {DRIFT_THRESHOLD * 100}%)...\n")

    # Features to monitor for distribution shifts
    monitored_features = ['experience_level', 'employment_type', 'company_size', 'remote_ratio']
    system_healthy = True

    # 3. Compute Drift Matrix
    for feature in monitored_features:
        shift_detected = calculate_categorical_drift(base_df, prod_df, feature)
        
        if shift_detected is not None:
            status = " [ OK ] "
            if shift_detected > DRIFT_THRESHOLD:
                status = "!!!! [ DRIFT DETECTED ] !!!!"
                system_healthy = False
            
            print(f"{status} {feature:<20} -> Max Shift: {shift_detected*100:.2f}%")

    print("\n--------------------------------------------------------------------------")
    if system_healthy:
        print(" SUCCESS: Production distributions are aligned with training parameters.")
        print(" System Status: HEALTHY")
    else:
        print(" WARNING: Substantial data drift detected in production input streams.")
        print(" Action Required: Retrain the predictive pipeline with recent log data.")
    print("==========================================================================\n")

if __name__ == "__main__":
    run_health_monitor()