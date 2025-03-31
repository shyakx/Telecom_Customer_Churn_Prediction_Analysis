import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import joblib

def load_data(filepath):
    """Load the dataset from CSV"""
    return pd.read_csv(filepath)

def preprocess_data(df, target_col='Churn'):
    """Preprocess the data and return train/test splits"""
    # Separate features and target
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    # Identify categorical and numerical columns
    cat_cols = X.select_dtypes(include=['object']).columns
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns
    
    # Create preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
        ])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    
    return X_train, X_test, y_train, y_test, preprocessor

def save_preprocessor(preprocessor, filepath):
    """Save the preprocessor object"""
    joblib.dump(preprocessor, filepath)

def load_preprocessor(filepath):
    """Load a saved preprocessor object"""
    return joblib.load(filepath)