"""
Training flow for wine dataset classification
"""
from metaflow import FlowSpec, step, Parameter
import os
import mlflow
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

class WineClassificationFlow(FlowSpec):
    """
    A flow for training a wine classification model using Metaflow
    """
    
    # Parameters for the flow
    random_state = Parameter('random_state', 
                             help='Random seed for reproducibility',
                             default=42,
                             type=int)
    
    test_size = Parameter('test_size',
                          help='Proportion of data to use for testing',
                          default=0.2,
                          type=float)
    
    cv_folds = Parameter('cv_folds',
                         help='Number of cross-validation folds',
                         default=5,
                         type=int)
    
    mlflow_tracking_uri = Parameter('mlflow_tracking_uri',
                                   help='MLflow tracking URI',
                                   default='http://localhost:5000',
                                   type=str)
    
    @step
    def start(self):
        """
        Load data and start the flow
        """
        # Import our data processing module
        import dataprocessing
        
        # Load and split the data
        self.train_data, self.test_data, self.train_labels, self.test_labels = (
            dataprocessing.load_data(test_size=self.test_size, 
                                    random_state=self.random_state)
        )
        
        print(f"Loaded wine dataset with {len(self.train_data)} training samples")
        print(f"and {len(self.test_data)} test samples")
        
        # Branch to different model training strategies
        self.next(self.train_knn, self.train_svm, self.train_rf)
    
    @step
    def train_knn(self):
        """
        Train a K-Nearest Neighbors classifier
        """
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.model_selection import GridSearchCV
        
        # Parameter grid for hyperparameter tuning
        param_grid = {
            'n_neighbors': [3, 5, 7, 9],
            'weights': ['uniform', 'distance'],
            'p': [1, 2]  # 1 for manhattan, 2 for euclidean
        }
        
        # Create base model
        base_model = KNeighborsClassifier()
        
        # Set up grid search with cross validation
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=self.cv_folds,
            scoring='accuracy',
            n_jobs=-1
        )
        
        # Fit the grid search to find best hyperparameters
        grid_search.fit(self.train_data, self.train_labels)
        
        # Get the best model
        self.model = grid_search.best_estimator_
        self.model_type = 'knn'
        self.best_params = grid_search.best_params_
        self.cv_score = grid_search.best_score_
        
        # Make predictions on test data
        self.predictions = self.model.predict(self.test_data)
        
        # Calculate metrics
        self.metrics = {
            'accuracy': accuracy_score(self.test_labels, self.predictions),
            'f1_score': f1_score(self.test_labels, self.predictions, average='weighted'),
            'precision': precision_score(self.test_labels, self.predictions, average='weighted'),
            'recall': recall_score(self.test_labels, self.predictions, average='weighted')
        }
        
        print(f"KNN best parameters: {self.best_params}")
        print(f"KNN CV accuracy: {self.cv_score:.4f}")
        print(f"KNN test accuracy: {self.metrics['accuracy']:.4f}")
        
        self.next(self.choose_model)

    @step
    def train_svm(self):
        """
        Train a Support Vector Machine classifier
        """
        from sklearn.svm import SVC
        from sklearn.model_selection import GridSearchCV
        
        # Parameter grid for hyperparameter tuning
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'kernel': ['linear', 'rbf', 'poly'],
            'gamma': ['scale', 'auto']
        }
        
        # Create base model
        base_model = SVC(probability=True)
        
        # Set up grid search with cross validation
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=self.cv_folds,
            scoring='accuracy',
            n_jobs=-1
        )
        
        # Fit the grid search to find best hyperparameters
        grid_search.fit(self.train_data, self.train_labels)
        
        # Get the best model
        self.model = grid_search.best_estimator_
        self.model_type = 'svm'
        self.best_params = grid_search.best_params_
        self.cv_score = grid_search.best_score_
        
        # Make predictions on test data
        self.predictions = self.model.predict(self.test_data)
        
        # Calculate metrics
        self.metrics = {
            'accuracy': accuracy_score(self.test_labels, self.predictions),
            'f1_score': f1_score(self.test_labels, self.predictions, average='weighted'),
            'precision': precision_score(self.test_labels, self.predictions, average='weighted'),
            'recall': recall_score(self.test_labels, self.predictions, average='weighted')
        }
        
        print(f"SVM best parameters: {self.best_params}")
        print(f"SVM CV accuracy: {self.cv_score:.4f}")
        print(f"SVM test accuracy: {self.metrics['accuracy']:.4f}")
        
        self.next(self.choose_model)
    
    @step
    def train_rf(self):
        """
        Train a Random Forest classifier
        """
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import GridSearchCV
        
        # Parameter grid for hyperparameter tuning
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5]
        }
        
        # Create base model
        base_model = RandomForestClassifier(random_state=self.random_state)
        
        # Set up grid search with cross validation
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=self.cv_folds,
            scoring='accuracy',
            n_jobs=-1
        )
        
        # Fit the grid search to find best hyperparameters
        grid_search.fit(self.train_data, self.train_labels)
        
        # Get the best model
        self.model = grid_search.best_estimator_
        self.model_type = 'random_forest'
        self.best_params = grid_search.best_params_
        self.cv_score = grid_search.best_score_
        
        # Make predictions on test data
        self.predictions = self.model.predict(self.test_data)
        
        # Calculate metrics
        self.metrics = {
            'accuracy': accuracy_score(self.test_labels, self.predictions),
            'f1_score': f1_score(self.test_labels, self.predictions, average='weighted'),
            'precision': precision_score(self.test_labels, self.predictions, average='weighted'),
            'recall': recall_score(self.test_labels, self.predictions, average='weighted')
        }
        
        print(f"Random Forest best parameters: {self.best_params}")
        print(f"Random Forest CV accuracy: {self.cv_score:.4f}")
        print(f"Random Forest test accuracy: {self.metrics['accuracy']:.4f}")
        
        self.next(self.choose_model)
    
    @step
    def choose_model(self, inputs):
        """
        Compare models and select the best one
        """
        # Create a list of models with their metrics
        models = []
        for inp in inputs:
            models.append({
                'model_type': inp.model_type,
                'model': inp.model,
                'cv_score': inp.cv_score,
                'test_accuracy': inp.metrics['accuracy'],
                'f1_score': inp.metrics['f1_score'],
                'best_params': inp.best_params
            })
        
        # Sort models by test accuracy
        sorted_models = sorted(models, key=lambda x: x['test_accuracy'], reverse=True)
        
        # Select the best model
        best_model = sorted_models[0]
        self.model = best_model['model']
        self.model_type = best_model['model_type']
        self.best_params = best_model['best_params']
        self.best_metrics = {
            'cv_score': best_model['cv_score'],
            'test_accuracy': best_model['test_accuracy'],
            'f1_score': best_model['f1_score']
        }
        
        # Save all results for reporting
        self.all_models = sorted_models
        
        # Explicitly set merged artifacts before calling merge_artifacts
        self.predictions = None  # We don't need to keep these
        self.metrics = None  # We're storing the best metrics in best_metrics
        self.cv_score = None  # We're storing this in best_metrics
        
        # Now we can safely merge the remaining artifacts
        self.merge_artifacts(inputs)
        
        print("\nBest Model:")
        print(f"Type: {self.model_type}")
        print(f"Parameters: {self.best_params}")
        print(f"CV Accuracy: {self.best_metrics['cv_score']:.4f}")
        print(f"Test Accuracy: {self.best_metrics['test_accuracy']:.4f}")
        print(f"F1 Score: {self.best_metrics['f1_score']:.4f}")
        
        self.next(self.register_model)
    
    @step
    def register_model(self):
        """
        Register the best model with MLflow
        """
        # Set up MLflow tracking
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        mlflow.set_experiment('wine-classification-metaflow')
        
        # Start MLflow run
        with mlflow.start_run(run_name=f'best-{self.model_type}-model'):
            # Log parameters
            mlflow.log_params(self.best_params)
            mlflow.log_param('model_type', self.model_type)
            mlflow.log_param('random_state', self.random_state)
            mlflow.log_param('test_size', self.test_size)
            mlflow.log_param('cv_folds', self.cv_folds)
            
            # Log metrics
            mlflow.log_metrics({
                'cv_accuracy': self.best_metrics['cv_score'],
                'test_accuracy': self.best_metrics['test_accuracy'],
                'f1_score': self.best_metrics['f1_score']
            })
            
            # Log and register the model
            mlflow.sklearn.log_model(
                self.model, 
                artifact_path='model',
                registered_model_name='wine-classification-model'
            )
            
            print(f"Model registered in MLflow as 'wine-classification-model'")
            
            # Get run ID for tracking
            self.mlflow_run_id = mlflow.active_run().info.run_id
        
        self.next(self.end)
    
    @step
    def end(self):
        """
        Finish the flow and display results
        """
        print("\nTraining Flow Complete")
        print(f"Best model: {self.model_type}")
        print(f"Test accuracy: {self.best_metrics['test_accuracy']:.4f}")
        print(f"MLflow run ID: {self.mlflow_run_id}")
        print("\nAll models comparison:")
        
        # Create a comparison table
        for i, model in enumerate(self.all_models):
            print(f"{i+1}. {model['model_type']} - Accuracy: {model['test_accuracy']:.4f}, CV: {model['cv_score']:.4f}")

if __name__ == '__main__':
    WineClassificationFlow() 