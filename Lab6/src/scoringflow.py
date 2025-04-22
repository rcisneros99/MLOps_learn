"""
Scoring flow for wine dataset classification
"""
from metaflow import FlowSpec, step, Parameter, JSONType, Flow
import mlflow
import numpy as np

class WineScoringFlow(FlowSpec):
    """
    A flow for scoring new samples using the best trained model
    """
    
    # Parameters
    sample = Parameter('sample',
                       help='JSON array of features for scoring',
                       type=JSONType,
                       required=True)
    
    model_name = Parameter('model_name',
                          help='Name of the registered model to use',
                          default='wine-classification-model',
                          type=str)
    
    model_stage = Parameter('model_stage',
                           help='Model stage to use (None, Staging, Production)',
                           default=None)
    
    mlflow_tracking_uri = Parameter('mlflow_tracking_uri',
                                   help='MLflow tracking URI',
                                   default='http://localhost:5000',
                                   type=str)
    
    use_latest_metaflow_model = Parameter('use_latest_metaflow_model',
                                         help='Whether to use the latest model from Metaflow',
                                         default=False,
                                         type=bool)
    
    @step
    def start(self):
        """
        Start the scoring flow and prepare the input data
        """
        # Convert input to numpy array
        self.input_data = np.array(self.sample).reshape(1, -1)
        
        # Validate input dimensions
        if self.input_data.shape[1] != 13:  # Wine dataset has 13 features
            raise ValueError(f"Expected 13 features, but got {self.input_data.shape[1]}")
        
        print(f"Input sample: {self.sample}")
        self.next(self.load_model)
    
    @step
    def load_model(self):
        """
        Load the model either from MLflow or from the latest Metaflow run
        """
        if self.use_latest_metaflow_model:
            # Load model from latest Metaflow training run
            train_run = Flow('WineClassificationFlow').latest_run
            self.model = train_run['register_model'].task.data.model
            self.model_type = train_run['register_model'].task.data.model_type
            self.mlflow_run_id = train_run['register_model'].task.data.mlflow_run_id
            
            print(f"Loaded {self.model_type} model from latest Metaflow run")
            print(f"Associated MLflow run ID: {self.mlflow_run_id}")
            
        else:
            # Load model from MLflow model registry
            mlflow.set_tracking_uri(self.mlflow_tracking_uri)
            
            # Load the model from the registry
            if self.model_stage:
                model_uri = f"models:/{self.model_name}/{self.model_stage}"
            else:
                model_uri = f"models:/{self.model_name}/latest"
            
            self.model = mlflow.sklearn.load_model(model_uri)
            self.model_source = model_uri
            
            print(f"Loaded model from MLflow: {model_uri}")
        
        self.next(self.make_prediction)
    
    @step
    def make_prediction(self):
        """
        Make predictions using the loaded model
        """
        # Make the prediction
        self.prediction = self.model.predict(self.input_data)[0]
        
        # Get class probabilities if available
        if hasattr(self.model, 'predict_proba'):
            self.probabilities = self.model.predict_proba(self.input_data)[0]
            self.has_probabilities = True
        else:
            self.has_probabilities = False
        
        print(f"Predicted class: {self.prediction}")
        
        # Map predictions to wine types (for better interpretation)
        wine_types = {
            0: 'class_0',
            1: 'class_1',
            2: 'class_2'
        }
        
        self.prediction_label = wine_types.get(self.prediction, f"Unknown class {self.prediction}")
        
        self.next(self.end)
    
    @step
    def end(self):
        """
        Summarize the prediction results
        """
        # Print prediction details
        print("\nWine Classification Results:")
        print(f"Prediction: {self.prediction} ({self.prediction_label})")
        
        # Print probability distribution if available
        if self.has_probabilities:
            print("\nClass Probabilities:")
            for i, prob in enumerate(self.probabilities):
                print(f"  Class {i}: {prob:.4f}")
        
        # Print model information
        if hasattr(self, 'model_type'):
            print(f"\nModel used: {self.model_type} from Metaflow")
        else:
            print(f"\nModel used: {self.model_source} from MLflow")
        
        # Create a simple results dictionary that could be returned by an API
        self.results = {
            'prediction': int(self.prediction),
            'prediction_label': self.prediction_label,
        }
        
        if self.has_probabilities:
            self.results['probabilities'] = self.probabilities.tolist()
        
        print("\nJSON result that could be returned by an API:")
        import json
        print(json.dumps(self.results, indent=2))

if __name__ == '__main__':
    WineScoringFlow() 