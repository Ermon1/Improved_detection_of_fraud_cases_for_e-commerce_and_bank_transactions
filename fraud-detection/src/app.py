import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import logging
import uvicorn

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="Fraud Detection API", description="API for detecting fraudulent transactions")

# Load Resources
model_pipeline = None

def load_resources():
    global model_pipeline
    base_dir = os.getcwd()
    models_dir = os.path.join(base_dir, 'models')
    model_path = os.path.join(models_dir, 'best_fraud_model.pkl')
    
    if os.path.exists(model_path):
        logger.info(f"Loading model from {model_path}...")
        model_pipeline = joblib.load(model_path)
    else:
        logger.error(f"Model not found at {model_path}")

# Load resources on startup
@app.on_event("startup")
async def startup_event():
    load_resources()

class PredictionRequest(BaseModel):
    # Accepting a list of dictionaries (records) or a single dictionary
    # For flexibility with the 195 features, we use Dict[str, Any]
    features: List[Dict[str, Any]]

class PredictionResponse(BaseModel):
    fraud_prediction: int
    probability: float

@app.post("/predict", response_model=List[PredictionResponse])
async def predict(request: PredictionRequest):
    if model_pipeline is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Convert input features to DataFrame
        # input data should match the columns expected by the model (processed data)
        df = pd.DataFrame(request.features)
        
        # Predict
        prediction = model_pipeline.predict(df)
        probability = model_pipeline.predict_proba(df)[:, 1]
        
        results = []
        for pred, prob in zip(prediction, probability):
            results.append({
                'fraud_prediction': int(pred),
                'probability': float(prob)
            })
            
        return results
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=5000)
