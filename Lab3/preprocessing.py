import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

def load_data():
    """
    Load the heart disease dataset from UCI repository
    """
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    column_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
    df = pd.read_csv(url, names=column_names, na_values='?')
    return df

def clean_data(df):
    """
    Clean the dataset by:
    1. Removing missing values
    2. Converting target to binary
    """
    # Remove rows with missing values
    df = df.dropna()
    
    # Convert target to binary (0 for no disease, 1 for disease)
    df['target'] = df['target'].map(lambda x: 1 if x > 0 else 0)
    
    return df

def split_and_scale_data(df, test_size=0.2, val_size=0.25, random_state=42):
    """
    Split the data into train, validation, and test sets
    Scale the features using StandardScaler
    """
    # Separate features and target
    X = df.drop('target', axis=1)
    y = df['target']
    
    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Second split: separate validation set from training set
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size, random_state=random_state
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    return (X_train_scaled, X_val_scaled, X_test_scaled, 
            y_train, y_val, y_test, scaler)

def save_processed_data(X_train, X_val, X_test, y_train, y_val, y_test, output_dir='data'):
    """
    Save the processed datasets
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the datasets
    np.save(f'{output_dir}/X_train_scaled.npy', X_train)
    np.save(f'{output_dir}/X_val_scaled.npy', X_val)
    np.save(f'{output_dir}/X_test_scaled.npy', X_test)
    np.save(f'{output_dir}/y_train.npy', y_train)
    np.save(f'{output_dir}/y_val.npy', y_val)
    np.save(f'{output_dir}/y_test.npy', y_test)

def main():
    """
    Main preprocessing pipeline
    """
    print("Loading data...")
    df = load_data()
    
    print("Cleaning data...")
    df_cleaned = clean_data(df)
    
    print("Splitting and scaling data...")
    (X_train_scaled, X_val_scaled, X_test_scaled,
     y_train, y_val, y_test, scaler) = split_and_scale_data(df_cleaned)
    
    print("Saving processed data...")
    save_processed_data(
        X_train_scaled, X_val_scaled, X_test_scaled,
        y_train, y_val, y_test
    )
    
    print("Preprocessing completed successfully!")

if __name__ == "__main__":
    main()
