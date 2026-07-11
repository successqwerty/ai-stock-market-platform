"""Pydantic schemas for request/response validation."""

from pydantic import BaseModel


class PredictionRequest(BaseModel):
    ticker: str = "AAPL"


class FeatureContribution(BaseModel):
    feature: str
    shap_value: float
    direction: str


class PredictionResponse(BaseModel):
    ticker: str
    prediction_date: str
    probability_up: float
    signal: str
    top_contributors: list[FeatureContribution]


class HealthResponse(BaseModel):
    status: str


class PriceHistoryPoint(BaseModel):
    date: str
    close: float
    volume: int


class PriceHistoryResponse(BaseModel):
    ticker: str
    history: list[PriceHistoryPoint]  