from fastapi import FastAPI, HTTPException
import uvicorn
from pydantic import BaseModel, Field
import pandas as pd
import joblib
import os
import json
from typing import Dict, Any, Union
import numpy as np
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = FastAPI(
    title="IoT Network Anomaly Detection API",
    description="API for detecting network anomalies in IoT devices",
    version="1.0.0"
)

class InputData(BaseModel):
    """Pydantic model for IoT network traffic input data."""
    
    Total_Fwd_Packets: float = Field(..., description="Total number of forward packets")
    Total_Backward_Packets: float = Field(..., description="Total number of backward packets")
    Total_Length_of_Fwd_Packets: float = Field(..., description="Total size of forward packets")
    Total_Length_of_Bwd_Packets: float = Field(..., description="Total size of backward packets")
    Flow_Duration: float = Field(..., description="Duration of the flow in microseconds")

    class Config:
        schema_extra = {
            "example": {
                "Total_Fwd_Packets": 10.0,
                "Total_Backward_Packets": 8.0,
                "Total_Length_of_Fwd_Packets": 1500.0,
                "Total_Length_of_Bwd_Packets": 1200.0,
                "Flow_Duration": 1000000.0
            }
        }

class ModelDeployment:
    def __init__(self, base_path: str = "models"):
        self.base_path = base_path
        os.makedirs(base_path, exist_ok=True)

    def load_latest_model(self):
        """Load the latest saved model and preprocessor."""
        try:
            model_folders = [f for f in os.listdir(self.base_path) if os.path.isdir(os.path.join(self.base_path, f))]
            if not model_folders:
                raise FileNotFoundError("No saved models found in the model directory.")

            model_folders.sort(reverse=True)
            latest_model_dir = os.path.join(self.base_path, model_folders[0])
            logging.info(f"Loading models from {latest_model_dir}")

            models = {}
            for f in os.listdir(latest_model_dir):
                if f.endswith('.joblib') and f != "preprocessor.joblib":
                    model_name = f.split('.')[0]
                    model_path = os.path.join(latest_model_dir, f)
                    models[model_name] = joblib.load(model_path)
                    logging.info(f"Loaded {model_name} model from {model_path}")

            preprocessor_path = os.path.join(latest_model_dir, "preprocessor.joblib")
            preprocessor = joblib.load(preprocessor_path)
            logging.info(f"Loaded preprocessor from {preprocessor_path}")

            return models, preprocessor
        except Exception as e:
            logging.error(f"Error loading models: {str(e)}")
            raise

@app.post("/predict")
async def predict(input_data: InputData, model_name: str = "random_forest"):
    """FastAPI endpoint for making predictions."""
    try:
        deployment = ModelDeployment()
        models, preprocessor = deployment.load_latest_model()
        
        model = models.get(model_name)
        if not model:
            raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found.")
            
        input_df = pd.DataFrame([input_data.dict()])
        
        logging.info(f"Feature names after preprocessing: {preprocessor.get_feature_names_out()}")
        
        X_input = preprocessor.transform(input_df)
        
        prediction_numeric = model.predict(X_input)
        prediction_proba = model.predict_proba(X_input)
        
        attack_type = preprocessor.label_encoder.inverse_transform(prediction_numeric)[0]
        
        class_probabilities = {
            label: float(prob) 
            for label, prob in zip(preprocessor.label_encoder.classes_, prediction_proba[0])
        }
        
        return {
            "prediction": attack_type,
            "confidence": float(max(prediction_proba[0])),
            "class_probabilities": class_probabilities,
            "model_used": model_name
        }
        
    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail="Prediction failed.")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
