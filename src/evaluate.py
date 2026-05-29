# src/evaluate.py
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class ModelEvaluator:
    @staticmethod
    def evaluate_predictions(y_true, y_pred, is_log_transformed=False):
        """
        Calculates MAE, RMSE, and R² scores. Automatically handles 
        exponential scaling if the target variable was log-transformed.
        """
        # Revert log transformation if applied during preprocessing
        if is_log_transformed:
            actual_true = np.expm1(y_true)
            actual_pred = np.expm1(y_pred)
        else:
            actual_true = y_true
            actual_pred = y_pred

        # Prevent negative predictions from breaking metrics down the line
        actual_pred = np.clip(actual_pred, a_min=0, a_max=None)

        # Compute core metrics
        mae = mean_absolute_error(actual_true, actual_pred)
        rmse = np.sqrt(mean_squared_error(actual_true, actual_pred))
        r2 = r2_score(actual_true, actual_pred)

        return {
            "MAE": round(mae, 2),
            "RMSE": round(rmse, 2),
            "R2_Score": round(r2, 4)
        }

    @staticmethod
    def generate_metrics_summary(trained_pipelines, X_test, y_test, is_log_transformed=False):
        """
        Iterates over all trained pipelines, generates predictions on the test 
        split, and returns a clean comparison DataFrame.
        """
        summary_data = {}

        for name, pipeline in trained_pipelines.items():
            # Generate inference predictions
            preds = pipeline.predict(X_test)
            
            # Extract scores
            metrics = ModelEvaluator.evaluate_predictions(y_test, preds, is_log_transformed)
            summary_data[name] = metrics

        # Format into an easily readable DataFrame table
        return pd.DataFrame(summary_data).T