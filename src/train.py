# src/train.py
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline

class SalaryModelTrainer:
    def __init__(self, preprocessor):
        """
        Initializes the trainer with a shared preprocessing pipeline
        and upgrades architectures to advanced tree ensembles.
        """
        self.preprocessor = preprocessor
        self.models = {
            "Random_Forest": RandomForestRegressor(
                n_estimators=150, 
                max_depth=10, 
                random_state=42, 
                n_jobs=-1
            ),
            "Gradient_Boosting": HistGradientBoostingRegressor(
                max_iter=150,
                max_depth=6,
                learning_rate=0.05,
                random_state=42
            )
        }
        self.trained_pipelines = {}

    def fit_all(self, X_train, y_train):
        """
        Binds the preprocessor with each ensemble architecture and executes training loops.
        """
        # HistGradientBoostingRegressor requires dense arrays, so we ensure standard fitting
        for name, model in self.models.items():
            print(f"[Engine] Deploying advanced training loop for: {name}...")
            
            full_pipeline = Pipeline(steps=[
                ('preprocessor', self.preprocessor),
                ('regressor', model)
            ])
            
            full_pipeline.fit(X_train, y_train)
            self.trained_pipelines[name] = full_pipeline
            
        print("[Engine] All advanced model architectures trained successfully.\n")
        return self.trained_pipelines