FROM python:3.9-slim

# Install mlflow
RUN pip install mlflow

# Expose the default mlflow port
EXPOSE 5000

# Command to run mlflow server
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000"] 