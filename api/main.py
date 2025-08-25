# api/main.py
import os
import pandas as pd
import mlflow
from fastapi import FastAPI
from pydantic import BaseModel

# Define the input data schema using Pydantic
class Passenger(BaseModel):
    Pclass: int
    Sex: int
    Age: float
    SibSp: int
    Parch: int
    Fare: float
    Embarked: int
    FamilySize: int
    IsAlone: int

# Set the MLflow tracking URI
# Make sure this is reachable from where you run the API
mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))

# Load the production model from the MLflow Model Registry
MODEL_NAME = "TitanicClassifier"
MODEL_STAGE = "Production"
model = mlflow.pyfunc.load_model(model_uri=f"models:/{MODEL_NAME}/{MODEL_STAGE}")

# Initialize the FastAPI app
app = FastAPI()

@app.get("/")
def read_root():
    return {"message": f"Titanic Survival Prediction API for model '{MODEL_NAME}' (Stage: {MODEL_STAGE})"}

@app.post("/predict")
def predict(passenger: Passenger):
    """
    Accepts passenger data and returns a survival prediction.
    """
    # Convert the input Pydantic model to a Pandas DataFrame
    input_df = pd.DataFrame([passenger.dict()])
    
    # Get the prediction from the loaded model
    prediction = model.predict(input_df)
    
    # Return the prediction as a JSON response
    return {
        "prediction": int(prediction[0]),
        "prediction_label": "Survived" if int(prediction[0]) == 1 else "Did not survive"
    }