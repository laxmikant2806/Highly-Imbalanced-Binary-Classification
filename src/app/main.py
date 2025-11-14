import torch
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib
from src.modeling.model import FraudDetector
from src.utils.utils import load_config
from src.modeling.predict import FraudDetectionPipeline

# Load configuration and initialize the pipeline
config = load_config()

# Load the trained model and scaler
pipeline_data = torch.load(config['model']['pipeline_save_path'])
model_state_dict = pipeline_data['model_state_dict']
scaler = pipeline_data['scaler']
threshold = pipeline_data['threshold']
model_architecture = pipeline_data['model_architecture']

model = FraudDetector(
    input_dim=model_architecture['input_dim'],
    hidden_dims=model_architecture['hidden_dims']
)
model.load_state_dict(model_state_dict)

pipeline = FraudDetectionPipeline(model, scaler, threshold)

# Initialize FastAPI app
app = FastAPI(title="Fraud Detection API")

# Define the request body
class Transaction(BaseModel):
    features: list

# Define the prediction endpoint
@app.post("/predict")
def predict(transaction: Transaction):
    """
    Predicts whether a transaction is fraudulent or not.

    - **features**: A list of 29 numerical features for the transaction.
    """
    features = np.array(transaction.features).reshape(1, -1)
    prediction = pipeline.predict(features)
    probability = pipeline.predict_proba(features)

    return {
        "is_fraud": int(prediction[0]),
        "fraud_probability": float(probability[0])
    }

# Health check endpoint
@app.get("/")
def read_root():
    return {"status": "ok"}
