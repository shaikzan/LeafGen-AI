
from contextlib import asynccontextmanager
import json
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.services.classifier import load_model
from app.services.generator import load_generator_model

from app.api.routes.predict import router as predict_router
from app.api.routes.generate import router as generate_router
from models.leafgen_model import classes, pretty


@asynccontextmanager
async def lifespan(app: FastAPI):

    print("Loading LeafGen AI models...")

    app.state.classifier = None
    app.state.generator = None

    # Load ResNet18 classifier
    app.state.classifier = load_model()
    print("ResNet18 loaded successfully!")

    # Load Conditional DCGAN
    app.state.generator = load_generator_model()
    print("Conditional DCGAN loaded successfully!")

    yield

    print("LeafGen AI backend shutting down...")


app = FastAPI(
    title="LeafGen AI API",
    description="Plant Disease Detection and Synthetic Leaf Generation",
    version="1.0.0",
    lifespan=lifespan
)

# Allow the local Vite development server to call the API. In production, set
# LEAFGEN_CORS_ORIGINS to a comma-separated list of deployed frontend origins.
cors_origins = os.getenv("LEAFGEN_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in cors_origins.split(",") if origin.strip()],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@app.get("/")
def home():
    return {
        "message": "Welcome to LeafGen AI API",
        "status": "Running"
    }


@app.get("/health")
def health_check():
    return {"message": "API process is running.", "status": "ok"}


@app.get("/ready")
def readiness_check():
    # Kept as a separate endpoint from liveness so orchestration can wait for models.
    if getattr(app.state, "classifier", None) is None or getattr(app.state, "generator", None) is None:
        raise HTTPException(status_code=503, detail="Models are not ready.")
    return {"status": "ready"}


@app.get("/api/classes")
def supported_classes():
    """Expose the checkpoint's authoritative index-to-class mapping to clients."""
    return {
        "count": len(classes),
        "classes": [
            {"index": index, "class_code": code, "class_name": pretty(code)}
            for index, code in enumerate(classes)
        ],
    }


@app.get("/api/results")
def experiment_results():
    """Return published run summaries; these are not recomputed at inference time."""
    results_path = Path(__file__).resolve().parents[1] / "models" / "results.json"
    if not results_path.is_file():
        raise HTTPException(status_code=503, detail="Published experiment results are not installed.")
    try:
        with results_path.open(encoding="utf-8") as file:
            return json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=500, detail="Published experiment results could not be read.") from exc


# Register disease prediction API
app.include_router(
    predict_router,
    prefix="/api",
    tags=["Disease Prediction"]
)


# Register synthetic generation API
app.include_router(
    generate_router,
    prefix="/api",
    tags=["Synthetic Generation"]
)
