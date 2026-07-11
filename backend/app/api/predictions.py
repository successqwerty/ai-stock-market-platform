"""API routes for predictions and price history."""

from fastapi import APIRouter, HTTPException

from backend.app.schemas.prediction import (
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
    PriceHistoryResponse,
)
from backend.app.services.prediction_service import get_price_history, predict_latest

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health_check():
    return {"status": "ok"}


@router.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    try:
        result = predict_latest(request.ticker)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/stocks/{ticker}/history", response_model=PriceHistoryResponse)
def stock_history(ticker: str, days: int = 30):
    try:
        result = get_price_history(ticker, days)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))