
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from app.services.classifier import load_model
from app.services.generator import load_generator_model

from app.api.routes.predict import router as predict_router
from app.api.routes.generate import router as generate_router


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
