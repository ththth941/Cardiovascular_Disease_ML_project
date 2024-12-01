from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import pandas as pd

app = FastAPI()

# Load model and scaler
model_filename = 'model/model_ML.pkl'
scaler_filename = 'model/mean_std_values_ML.pkl'

with open(model_filename, 'rb') as file:
    model = pickle.load(file)

with open(scaler_filename, 'rb') as file:
    mean_std_values = pickle.load(file)


# Define input schema
class HeartDiseaseInput(BaseModel):
    age: int
    gender: int
    chestpain: int
    restingBP: int
    serumcholestrol: int
    fastingbloodsugar: int
    restingrelectro: int
    maxheartrate: int
    exerciseangia: int
    oldpeak: float
    slope: int
    noofmajorvessels: int


@app.post("/predict")
async def predict(input_data: HeartDiseaseInput):
    # Convert input data to DataFrame
    data = pd.DataFrame([input_data.dict()])

    # Apply mean and std normalization
    data = (data - mean_std_values['mean']) / mean_std_values['std']

    # Perform prediction
    prediction = model.predict(data)
    prediction_proba = model.predict_proba(data)

    # Prepare response
    if prediction[0] == 1:
        result = "Positive"
        confidence = prediction_proba[0][1]
    else:
        result = "Negative"
        confidence = prediction_proba[0][0]

    return {
        "prediction": result,
        "confidence": round(confidence * 100, 2)
    }
