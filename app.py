from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

model = joblib.load("model.pkl")
app = FastAPI()

class PatientData(BaseModel):
    mean_radius: float
    mean_texture: float
    mean_perimeter: float
    mean_area: float
    mean_smoothness: float
    mean_compactness: float
    mean_concavity: float
    mean_concave_points: float
    mean_symmetry: float
    mean_fractal_dimension: float
    radius_error: float
    texture_error: float
    perimeter_error: float
    area_error: float
    smoothness_error: float
    compactness_error: float
    concavity_error: float
    concave_points_error: float
    symmetry_error: float
    fractal_dimension_error: float
    worst_radius: float
    worst_texture: float
    worst_perimeter: float
    worst_area: float
    worst_smoothness: float
    worst_compactness: float
    worst_concavity: float
    worst_concave_points: float
    worst_symmetry: float
    worst_fractal_dimension: float

@app.get("/")
def read_root():
    return {"message": "Weekly retrained model API is live."}

@app.post("/predict")
def predict(patient: PatientData):
    input_data = np.array([[
        patient.mean_radius, patient.mean_texture, patient.mean_perimeter,
        patient.mean_area, patient.mean_smoothness, patient.mean_compactness,
        patient.mean_concavity, patient.mean_concave_points, patient.mean_symmetry,
        patient.mean_fractal_dimension, patient.radius_error, patient.texture_error,
        patient.perimeter_error, patient.area_error, patient.smoothness_error,
        patient.compactness_error, patient.concavity_error, patient.concave_points_error,
        patient.symmetry_error, patient.fractal_dimension_error, patient.worst_radius,
        patient.worst_texture, patient.worst_perimeter, patient.worst_area,
        patient.worst_smoothness, patient.worst_compactness, patient.worst_concavity,
        patient.worst_concave_points, patient.worst_symmetry, patient.worst_fractal_dimension
    ]])
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0].max()
    result = "Benign (Low Risk)" if prediction == 1 else "Malignant (High Risk)"
    return {"prediction": result, "confidence": round(float(probability), 4)}
