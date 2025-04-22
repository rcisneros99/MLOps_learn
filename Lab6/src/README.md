# Wine Classification with Metaflow

This project demonstrates how to use Metaflow to create ML flows for training and inference on the wine dataset.

## Project Structure

- `dataprocessing.py`: Module for loading and preprocessing the wine dataset
- `trainingflow.py`: Metaflow pipeline for training and comparing multiple models
- `scoringflow.py`: Metaflow pipeline for making predictions with the trained model

## Training Flow

The training flow performs the following steps:

1. Load and split the wine dataset
2. Train three different models in parallel:
   - K-Nearest Neighbors (KNN)
   - Support Vector Machine (SVM)
   - Random Forest
3. Compare model performance and select the best model
4. Register the best model in MLflow

### Running the Training Flow

```bash
python trainingflow.py run --cv_folds 3 --mlflow_tracking_uri http://127.0.0.1:5000
```

Parameters:
- `random_state`: Random seed for reproducibility (default: 42)
- `test_size`: Proportion of data to use for testing (default: 0.2)
- `cv_folds`: Number of cross-validation folds (default: 5)
- `mlflow_tracking_uri`: MLflow tracking URI (default: http://localhost:5000)

## Scoring Flow

The scoring flow performs the following steps:

1. Load input data for prediction
2. Load the model (either from MLflow registry or latest Metaflow run)
3. Make a prediction and return results

### Running the Scoring Flow

```bash
# Using the model from MLflow (default)
python scoringflow.py run --sample '[14.3,1.92,2.72,20.0,120.0,2.8,3.14,0.33,1.97,6.2,1.07,2.65,1280.0]' --mlflow_tracking_uri http://127.0.0.1:5000

# Using the model from the latest Metaflow run
python scoringflow.py run --sample '[14.3,1.92,2.72,20.0,120.0,2.8,3.14,0.33,1.97,6.2,1.07,2.65,1280.0]' --use_latest_metaflow_model True
```

Parameters:
- `sample`: JSON array of features for scoring (required)
- `model_name`: Name of the registered MLflow model (default: wine-classification-model)
- `model_stage`: MLflow model stage (None, Staging, Production) (default: None)
- `mlflow_tracking_uri`: MLflow tracking URI (default: http://localhost:5000)
- `use_latest_metaflow_model`: Whether to use the latest model from Metaflow (default: False)

## MLflow Integration

The training flow integrates with MLflow by:

1. Creating a new experiment if it doesn't exist
2. Logging model parameters, metrics, and artifacts
3. Registering the best model in the MLflow Model Registry

You can access the MLflow UI at [http://127.0.0.1:5000](http://127.0.0.1:5000) when running the MLflow server. 