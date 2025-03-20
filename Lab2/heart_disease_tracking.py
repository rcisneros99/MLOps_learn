# Import required libraries
import mlflow
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
from collections import defaultdict

# ML models we'll use
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Dictionary to store model performances
model_performances = defaultdict(dict)

# Set up MLflow tracking
mlflow.set_tracking_uri('sqlite:///mlflow.db')
mlflow.set_experiment('heart-disease-classification')

# Load the dataset from a reliable source
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
column_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
df = pd.read_csv(url, names=column_names, na_values='?')

# Clean the data
df = df.dropna()  # Remove rows with missing values
df['target'] = df['target'].map(lambda x: 1 if x > 0 else 0)  # Convert target to binary

# Basic data exploration
print("Dataset Shape:", df.shape)
print("\nMissing Values:")
print(df.isnull().sum())
print("\nData Types:")
print(df.dtypes)

# Create visualization directory
os.makedirs('visualizations', exist_ok=True)

# Create correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', center=0)
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.savefig('visualizations/correlation_heatmap.png')
plt.close()

# Prepare data for modeling
X = df.drop('target', axis=1)
y = df['target']

# Split data into train, validation, and test sets
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Save the datasets
os.makedirs('data', exist_ok=True)
np.save('data/X_train_scaled.npy', X_train_scaled)
np.save('data/X_val_scaled.npy', X_val_scaled)
np.save('data/X_test_scaled.npy', X_test_scaled)
np.save('data/y_train.npy', y_train)
np.save('data/y_val.npy', y_val)
np.save('data/y_test.npy', y_test)

def log_model_performance(model_name, model, X_train, X_val, y_train, y_val):
    with mlflow.start_run(run_name=model_name) as run:
        # Log model parameters
        mlflow.log_params(model.get_params())
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)
        
        # Calculate metrics
        train_accuracy = accuracy_score(y_train, y_train_pred)
        val_accuracy = accuracy_score(y_val, y_val_pred)
        val_precision = precision_score(y_val, y_val_pred)
        val_recall = recall_score(y_val, y_val_pred)
        val_f1 = f1_score(y_val, y_val_pred)
        
        # Store performance metrics
        model_performances[run.info.run_id] = {
            'model_name': model_name,
            'val_accuracy': val_accuracy,
            'val_precision': val_precision,
            'val_recall': val_recall,
            'val_f1': val_f1,
            'run_id': run.info.run_id
        }
        
        # Log metrics
        mlflow.log_metric("train_accuracy", train_accuracy)
        mlflow.log_metric("val_accuracy", val_accuracy)
        mlflow.log_metric("val_precision", val_precision)
        mlflow.log_metric("val_recall", val_recall)
        mlflow.log_metric("val_f1", val_f1)
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        # Log datasets
        mlflow.log_artifacts("data")
        
        return run.info.run_id

# Try different C values for Logistic Regression
print("\nTraining Logistic Regression models...")
for C in [0.01, 0.1, 1.0, 10.0]:
    model = LogisticRegression(C=C, random_state=42, max_iter=1000)
    log_model_performance(f"LogisticRegression_C{C}", model, 
                         X_train_scaled, X_val_scaled, y_train, y_val)

# Try different max_depths for Decision Tree
print("\nTraining Decision Tree models...")
for max_depth in [3, 5, 7, 10]:
    model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    log_model_performance(f"DecisionTree_depth{max_depth}", model,
                         X_train_scaled, X_val_scaled, y_train, y_val)

# Try different combinations for Random Forest
print("\nTraining Random Forest models...")
for n_estimators in [50, 100, 200]:
    for max_depth in [5, 10]:
        model = RandomForestClassifier(n_estimators=n_estimators, 
                                     max_depth=max_depth,
                                     random_state=42)
        log_model_performance(f"RandomForest_trees{n_estimators}_depth{max_depth}", 
                            model, X_train_scaled, X_val_scaled, y_train, y_val)

# Feature Selection
print("\nPerforming feature selection...")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train)

# Get feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf.feature_importances_
})
feature_importance = feature_importance.sort_values('importance', ascending=False)

# Plot feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importance)
plt.title('Feature Importance')
plt.tight_layout()
plt.savefig('visualizations/feature_importance.png')
plt.close()

# Select top 8 features
top_features = feature_importance['feature'].head(8).values
X_train_selected = X_train_scaled[:, [X.columns.get_loc(col) for col in top_features]]
X_val_selected = X_val_scaled[:, [X.columns.get_loc(col) for col in top_features]]
X_test_selected = X_test_scaled[:, [X.columns.get_loc(col) for col in top_features]]

# Train models with selected features
print("\nTraining models with selected features...")
# Logistic Regression
model = LogisticRegression(C=1.0, random_state=42, max_iter=1000)
log_model_performance("LogisticRegression_SelectedFeatures", model,
                     X_train_selected, X_val_selected, y_train, y_val)

# Decision Tree
model = DecisionTreeClassifier(max_depth=5, random_state=42)
log_model_performance("DecisionTree_SelectedFeatures", model,
                     X_train_selected, X_val_selected, y_train, y_val)

# Random Forest (Final Model)
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
final_run_id = log_model_performance("RandomForest_SelectedFeatures", model,
                                   X_train_selected, X_val_selected, y_train, y_val)

# Evaluate Final Model on Test Set
print("\nEvaluating final model on test set...")
with mlflow.start_run(run_name="FinalModel_TestSet"):
    best_model = mlflow.sklearn.load_model(f"runs:/{final_run_id}/model")
    
    # Make predictions on test set
    y_test_pred = best_model.predict(X_test_selected)
    
    # Calculate and log test metrics
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred)
    test_recall = recall_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred)
    
    print(f"\nTest Set Metrics:")
    print(f"Accuracy: {test_accuracy:.4f}")
    print(f"Precision: {test_precision:.4f}")
    print(f"Recall: {test_recall:.4f}")
    print(f"F1 Score: {test_f1:.4f}")
    
    mlflow.log_metric("test_accuracy", test_accuracy)
    mlflow.log_metric("test_precision", test_precision)
    mlflow.log_metric("test_recall", test_recall)
    mlflow.log_metric("test_f1", test_f1)
    
    # Register the model
    mlflow.sklearn.log_model(best_model, "production_model",
                            registered_model_name="HeartDiseaseClassifier")

# After all models are trained, find the top 3 models
print("\nIdentifying top 3 models...")
model_results = pd.DataFrame.from_dict(model_performances, orient='index')
top_3_models = model_results.nlargest(3, 'val_f1')

print("\nTop 3 Models by Validation F1 Score:")
print("=====================================")
for idx, row in top_3_models.iterrows():
    print(f"Model: {row['model_name']}")
    print(f"Validation F1 Score: {row['val_f1']:.4f}")
    print(f"Validation Accuracy: {row['val_accuracy']:.4f}")
    print(f"Run ID: {row['run_id']}")
    print("-------------------------------------")

# Save top 3 models information
top_3_models.to_csv('top_3_models.csv') 