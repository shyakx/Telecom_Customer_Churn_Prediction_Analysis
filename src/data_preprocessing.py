import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_data(df):
    """Preprocess the telecom churn data for modeling"""
    # Convert target to binary
    df['Churn'] = df['Churn'].map({True: 1, False: 0})
    
    # Convert categorical features
    df['International plan'] = df['International plan'].map({'Yes': 1, 'No': 0})
    df['Voice mail plan'] = df['Voice mail plan'].map({'Yes': 1, 'No': 0})
    
    # Drop unnecessary columns
    df = df.drop(['State', 'Area code'], axis=1)
    
    # Separate features and target
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    return X, y