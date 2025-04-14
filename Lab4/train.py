import mlflow
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from preprocessing import load_data, clean_data, split_and_scale_data

def train_and_log_model(X_train, X_val, y_train, y_val, params):
    with mlflow.start_run():
        # Log parameters
        for param_name, param_value in params.items():
            mlflow.log_param(param_name, param_value)
        
        # Train model
        model = LogisticRegression(**params)
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_val)
        
        # Calculate metrics
        accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred)
        recall = recall_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)
        
        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        return model, accuracy, precision, recall, f1

def main():
    # Set MLflow tracking URI to our Kubernetes MLflow server
    mlflow.set_tracking_uri('http://127.0.0.1:5000')
    mlflow.set_experiment("heart-disease-experiment")
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    df = load_data()
    df_cleaned = clean_data(df)
    X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test, scaler = split_and_scale_data(df_cleaned)
    
    # Define parameters for logistic regression
    params = {
        "C": 1.0,
        "max_iter": 1000,
        "random_state": 42
    }
    
    # Train and log the model
    print("Training model and logging to MLflow...")
    model, accuracy, precision, recall, f1 = train_and_log_model(
        X_train_scaled, X_val_scaled, y_train, y_val, params
    )
    
    print("\nModel Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

if __name__ == "__main__":
    main() 