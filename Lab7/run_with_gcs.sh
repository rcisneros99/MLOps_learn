#!/bin/bash

# This script uses Google Cloud Storage instead of AWS S3 for Metaflow
# You need to have Google Cloud SDK (gcloud) installed and configured
export METAFLOW_KUBERNETES_NAMESPACE=default
# Step 1: Check if gcloud is installed and authenticated
if ! command -v gcloud &> /dev/null; then
    echo "Google Cloud SDK (gcloud) is not installed. Please install it first."
    echo "Visit: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Verify authentication
echo "Checking Google Cloud authentication..."
gcloud auth list

# Step 2: Set up a GCS bucket
# Replace with your preferred bucket name
GCS_BUCKET="metaflow-$(whoami)-$(date +%s)"
GCS_PROJECT=$(gcloud config get-value project)

if [ -z "$GCS_PROJECT" ]; then
    echo "No Google Cloud project is set. Please run 'gcloud config set project YOUR_PROJECT_ID'"
    exit 1
fi

echo "Creating GCS bucket $GCS_BUCKET in project $GCS_PROJECT..."
gsutil mb -p $GCS_PROJECT gs://$GCS_BUCKET || true

# Step 3: Configure Metaflow to use GCS
export METAFLOW_DATASTORE_SYSROOT_GS="gs://$GCS_BUCKET/metaflow"
export METAFLOW_DEFAULT_DATASTORE=gs
export METAFLOW_DEFAULT_METADATA=local

# Step 4: Install necessary packages
pip install kubernetes google-cloud-storage

# Step 5: Run the Metaflow flow
# Update the file to use Google Cloud Storage
echo "Modifying flow to use GCS..."

cat > gcs_kubernetes_flow.py << EOF
from metaflow import FlowSpec, step, kubernetes

class ClassifierTrainFlow(FlowSpec):

    @step
    def start(self):
        from sklearn import datasets
        from sklearn.model_selection import train_test_split
        import numpy as np

        X, y = datasets.load_wine(return_X_y=True)
        self.train_data, self.test_data, self.train_labels, self.test_labels = train_test_split(X,y, test_size=0.2, random_state=0)
        print("Data loaded successfully")
        # Use fewer lambdas for faster execution
        self.lambdas = np.arange(0.1, 1, 0.2)
        self.next(self.train_lasso, foreach='lambdas')

    @kubernetes(
        cpu=1,
        memory=4096,
        image="python:3.9"
    )
    @step
    def train_lasso(self):
        from sklearn.linear_model import Lasso

        self.model = Lasso(alpha=self.input)
        self.model.fit(self.train_data, self.train_labels)
        self.next(self.choose_model)

    @step
    def choose_model(self, inputs):
        def score(inp):
            return inp.model, inp.model.score(inp.test_data, inp.test_labels)

        self.results = sorted(map(score, inputs), key=lambda x: -x[1])
        self.model = self.results[0][0]
        self.next(self.end)

    @step
    def end(self):
        print('Scores:')
        print('\n'.join('%s %f' % res for res in self.results))
        print('Model:', self.model)


if __name__=='__main__':
    ClassifierTrainFlow()
EOF

echo "Running Metaflow with GCS and Kubernetes..."
python gcs_kubernetes_flow.py --datastore=gs run

echo "Setup complete!"
echo "Your GCS bucket is: gs://$GCS_BUCKET"
echo "To clean up when done, run: gsutil rm -r gs://$GCS_BUCKET" 