"""
FastAPI app to serve the trained model.
"""
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"msg": "Titanic Survival Prediction API is running 🚀"}
