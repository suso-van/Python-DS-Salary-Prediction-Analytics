# src/pipeline.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def clean_job_titles(title):
    """
    Consolidates 50+ chaotic job titles into 5 core high-level domains
    to eliminate high-cardinality variance.
    """
    title = str(title).lower()
    if 'machine learning' in title or 'ml' in title or 'computer vision' in title or 'nlp' in title:
        return 'ML Engineer'
    elif 'data scientist' in title or 'science' in title or 'ai' in title or 'research' in title:
        return 'Data Scientist'
    elif 'engineer' in title or 'architect' in title or 'developer' in title:
        return 'Data Engineer'
    elif 'analyst' in title or 'bi' in title or 'analytics' in title:
        return 'Data Analyst'
    elif 'manager' in title or 'head' in title or 'director' in title or 'lead' in title:
        return 'Data Leadership'
    else:
        return 'Other Tech Specialist'

def create_preprocessing_pipeline(numeric_features, categorical_features, ordinal_features, mapping_orders):
    """
    Constructs an isolated, scalable data processing pipeline matching the 
    Kaggle Data Science Salaries schema.
    """
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    ordinal_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ordinal', OrdinalEncoder(categories=mapping_orders))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features),
            ('ord', ordinal_transformer, ordinal_features)
        ],
        remainder='drop'
    )
    
    return preprocessor

def load_and_prep_data(data_path, target_col='salary_in_usd', test_size=0.2, random_state=42):
    """
    Loads the real Kaggle CSV, runs feature engineering to consolidate roles, 
    and handles train-test separation.
    """
    df = pd.read_csv(data_path)
    
    if 'X' in df.columns:
        df = df.drop(columns=['X'])
        
    # Consolidate job titles BEFORE splitting to guarantee clean categorization
    df['job_title'] = df['job_title'].apply(clean_job_titles)
    
    leakage_cols = ['salary', 'salary_currency']
    df = df.drop(columns=[col for col in leakage_cols if col in df.columns])
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    is_skewed = False
    if y.skew() > 1.0:
        y = np.log1p(y)
        is_skewed = True
        
    return train_test_split(X, y, test_size=test_size, random_state=random_state), is_skewed