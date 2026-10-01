from fastapi import FastAPI
from app.schemas import Measurement
from app.features import build_features
from app.model import FuelTheftPredictor
from app.database import get_vehicle_history, insert_measurement

app = FastAPI(
    title="Fuel Theft Detection API",
    description="API for real-time fuel event detection using Random Forest + MLP stacked model.",
    version="1.0.0"
)

predictor = FuelTheftPredictor(artifact_dir="artifacts")


@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "Fuel Theft Detection API is running"
    }


@app.post("/predict")
def predict(measurement: Measurement):
    current = measurement.model_dump()

    history = get_vehicle_history(
        obj_id=current["obj_id"],
        current_time=current["time"]
    )

    # Prvo mjerenje novog vozila nema prethodni kontekst.
    # Zato ga samo spremamo u bazu, ali ne šaljemo u model.
    if len(history) == 0:
        insert_measurement(current)

        return {
            "prediction_class": None,
            "prediction_label": "insufficient_history",
            "confidence": 0.0,
            "probabilities": {},
            "message": "First measurement stored. Prediction requires previous vehicle history."
        }

    features = build_features(
        current_measurement=current,
        history=history
    )

    result = predictor.predict(features)

    insert_measurement(current)

    return result