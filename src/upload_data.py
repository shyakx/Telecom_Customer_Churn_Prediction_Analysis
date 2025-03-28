import pandas as pd
import os

def handle_upload(file):
    """Validate and save uploaded CSV file"""
    # Check file extension
    if not file.filename.endswith('.csv'):
        raise ValueError('Only CSV files are accepted')
    
    # Read the file
    try:
        df = pd.read_csv(file)
    except Exception as e:
        raise ValueError('Invalid CSV file format')
    
    # Validate columns
    required_columns = {
        'State', 'Account length', 'Area code', 'International plan',
        'Voice mail plan', 'Number vmail messages', 'Total day minutes',
        'Total day calls', 'Total day charge', 'Total eve minutes',
        'Total eve calls', 'Total eve charge', 'Total night minutes',
        'Total night calls', 'Total night charge', 'Total intl minutes',
        'Total intl calls', 'Total intl charge', 'Customer service calls',
        'Churn'
    }
    
    if not required_columns.issubset(set(df.columns)):
        missing = required_columns - set(df.columns)
        raise ValueError(f'Missing required columns: {missing}')
    
    # Save the file
    os.makedirs('data/uploaded', exist_ok=True)
    save_path = os.path.join('data/uploaded', file.filename)
    df.to_csv(save_path, index=False)
    
    return {
        'message': 'File uploaded successfully',
        'rows_received': len(df),
        'columns_received': list(df.columns)
    }