from pydantic import BaseModel


class Measurement(BaseModel):
    obj_id: str
    time: str
    fuel_lvl: float
    speed: float
    contact_value: int
    gps_latitude: float
    gps_longitude: float


class PredictionResponse(BaseModel):
    prediction_class: int
    prediction_label: str
    confidence: float
    probabilities: dict[str, float]