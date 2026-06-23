
from fastapi import FastAPI
from pydantic import BaseModel, Field
import numpy as np


import joblib
model = joblib.load("egfr_stage_model.pkl")



app = FastAPI()

class InputData(BaseModel):
    features: list

# 1. Define what the success response SHOULD look like
# Change the field name here
class PredictionOutput(BaseModel):
    prediction: list = Field(..., alias="eGFR prediction") 
    # Or simply:
    # eGFR_prediction: list

# 2. Add 'response_model' to your post decorator
@app.post("/predict", response_model=PredictionOutput)
def predict(data: InputData):
    # Your model prediction logic
    raw_prediction = model.predict([data.features])
    
    # Convert numpy array to list so it can be sent as JSON
    prediction_list = raw_prediction.tolist() if hasattr(raw_prediction, 'tolist') else list(raw_prediction)
    
    # Return it matching the PredictionOutput structure
    return {"eGFR prediction": prediction_list}