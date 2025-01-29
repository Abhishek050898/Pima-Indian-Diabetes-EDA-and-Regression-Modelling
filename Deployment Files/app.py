from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import uvicorn
import mlflow.sklearn
import numpy as np
import os
import pickle


# mlflow.set_tracking_uri("http://localhost:5000")


# Picking the best model from mlflow
# best_run_id = "d376b4b3f8b9441d98ba4b7c27040555"
# model_uri = f"runs:/{best_run_id}"
# model = mlflow.sklearn.load_model(model_uri)

# Load the model directly from the file system
MODEL_PATH = "model.pkl"

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    print("✅ Model loaded successfully!")
except FileNotFoundError:
    print(f"❌ Error: Model file '{MODEL_PATH}' not found!")

app = FastAPI()

templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html",{"request":request})

@app.post("/predict/")
async def predict(
    request: Request,
    Pregnancies: int = Form(...),
    Glucose: int = Form(...),
    BloodPressure: int = Form(...),
    SkinThickness: int = Form(...),
    Insulin: int = Form(...),
    BMI: float = Form(...),
    DiabetesPedigreeFunction: float = Form(...),
    Age: int = Form(...)
):
    input_features = np.array([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]])

    prediction = model.predict(input_features)
    outcome = "Diabetic" if prediction[0]== 1 else "Non-Diabetic"

    return templates.TemplateResponse("index.html", {"request": request, "outcome": outcome})

    # return {
    #     "Pregnancies": Pregnancies,
    #     "Glucose": Glucose,
    #     "BloodPressure": BloodPressure,
    #     "SkinThickness": SkinThickness,
    #     "Insulin": Insulin,
    #     "BMI": BMI,
    #     "DiabetesPedigreeFunction": DiabetesPedigreeFunction,
    #     "Age": Age,
    #     "Prediction": outcome
    # }


# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)
