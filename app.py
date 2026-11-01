
from contextlib import asynccontextmanager
from pathlib import Path

import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models" / "champion"

FEATURE_COLUMNS = [
    "precipitation",
    "soil_moisture",
    "river_discharge_lag1",
    "river_discharge_lag2",
    "river_discharge_lag3",
    "precipitation_roll3",
    "precipitation_roll7",
    "soil_moisture_roll3",
    "month",
    "is_rainy_season",
]

model = None


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    precipitation: float = Field(ge=0)
    soil_moisture: float = Field(ge=0, le=1)

    river_discharge_lag1: float = Field(ge=0)
    river_discharge_lag2: float = Field(ge=0)
    river_discharge_lag3: float = Field(ge=0)

    precipitation_roll3: float = Field(ge=0)
    precipitation_roll7: float = Field(ge=0)
    soil_moisture_roll3: float = Field(ge=0, le=1)

    month: int = Field(ge=1, le=12)
    is_rainy_season: int = Field(ge=0, le=1)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model

    model = mlflow.pyfunc.load_model(str(MODEL_DIR))
    print("Model Champion berhasil dimuat")

    yield

    model = None


app = FastAPI(
    title="Prediksi Debit Sungai API",
    description="REST API untuk inferensi model Champion LK-07",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
def root():
    return {
        "message": "Prediksi Debit Sungai API",
        "status": "running",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy" if model is not None else "unavailable",
        "model_loaded": model is not None,
    }


@app.post("/predict")
def predict(payload: PredictionRequest):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model belum tersedia",
        )

    values = payload.model_dump()

    input_df = pd.DataFrame(
        [[values[column] for column in FEATURE_COLUMNS]],
        columns=FEATURE_COLUMNS,
    )

    try:
        prediction = model.predict(input_df)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Gagal menjalankan prediksi",
        ) from exc

    return {
        "status": "success",
        "prediction": float(prediction[0]),
        "unit": "m3/s",
        "model": "debit-sungai-regressor",
    }
