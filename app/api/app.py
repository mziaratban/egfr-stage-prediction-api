from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import os
# for monitoring
from prometheus_fastapi_instrumentator import Instrumentator
from .middleware import PredictionMetricsMiddleware
from .metrics import (
    prediction_total,
    prediction_stage_counter,
    prediction_failures,
)



app = FastAPI(title="eGFR Prediction API", version="1.0")

app.add_middleware(PredictionMetricsMiddleware)

Instrumentator().instrument(app).expose(app)






# -----------------------
# Model path (IMPORTANT)
# -----------------------
MODEL_DIR = "/app/app/models"


def load_model(name):
    return joblib.load(os.path.join(MODEL_DIR, name))


model_1 = load_model("egfr_stage_model_day_1.pkl")
model_2 = load_model("egfr_stage_model_day_2.pkl")
model_3 = load_model("egfr_stage_model_day_3.pkl")
model_4 = load_model("egfr_stage_model_day_4.pkl")
model_5 = load_model("egfr_stage_model_day_5.pkl")
model_6 = load_model("egfr_stage_model_day_6.pkl")
model_7 = load_model("egfr_stage_model_day_7.pkl")


# -----------------------
# Schema
# -----------------------
class InputData(BaseModel):
    features: list[float]


class PredictionOutput(BaseModel):
    predictions: list[int]


# -----------------------
# Health
# -----------------------
@app.get("/")
def root():
    return {"status": "running"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/version")
def version():
    return {"version": "2026-06-22-v1"}


# -----------------------
# Predict
# -----------------------
@app.post("/predict", response_model=PredictionOutput)
def predict(data: InputData):

    try:
        prediction_total.inc()

        x = np.array(data.features, dtype=float).reshape(1, -1)

        preds = [
            5-int(model_1.predict(x)[0]),
            5-int(model_2.predict(x)[0]),
            5-int(model_3.predict(x)[0]),
            5-int(model_4.predict(x)[0]),
            5-int(model_5.predict(x)[0]),
            5-int(model_6.predict(x)[0]),
            5-int(model_7.predict(x)[0]),
        ]

        
        for day, stage in enumerate(preds, start=1):

            prediction_stage_counter.labels(
                day=str(day),
                stage=str(stage)
            ).inc()
    
    
        return {"predictions": preds}


    except Exception:

        prediction_failures.inc()

        raise