
import sys
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

from fastapi import HTTPException

BACKEND_DIR = Path(__file__).resolve().parents[2]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from models.leafgen_model import load_classifier, classify, pretty


def load_model():
    """Load the trained ResNet18 classifier."""
    return load_classifier(variant="real")


def predict_image(model, image_bytes: bytes):
    """Predict disease from uploaded image bytes."""

    try:
        with Image.open(BytesIO(image_bytes)) as source_image:
            if source_image.width * source_image.height > 25_000_000:
                raise HTTPException(status_code=413, detail="Image dimensions are too large.")
            image = source_image.convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        raise HTTPException(
            status_code=400,
            detail="Invalid image file."
        )

    # Convert image into RGB NumPy array
    rgb_image = np.asarray(image, dtype=np.uint8)

    # Get top 3 predictions
    predictions = classify(model, rgb_image, k=3)

    # Format predictions
    results = [
        {
            "class_name": pretty(class_name),
            "class_code": class_name,
            "confidence": round(probability, 4)
        }
        for class_name, probability in predictions
    ]

    top = results[0]

    return {
        "success": True,
        "prediction": top["class_name"],
        "confidence": top["confidence"],
        "top_predictions": results,
        "warning": (
            "Low confidence. Please upload a clearer leaf image."
            if top["confidence"] < 0.70
            else None
        )
    }
