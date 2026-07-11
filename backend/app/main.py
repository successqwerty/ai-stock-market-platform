"""FastAPI application entry point."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.predictions import router

app = FastAPI(
    title="AI Stock Market Intelligence Platform API",
    description="Serves ML-driven direction predictions and explanations for research purposes only. Not financial advice.",
    version="0.1.0",
)

# Allow a frontend (e.g. running on localhost:3000 or similar) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # permissive for local development only
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)