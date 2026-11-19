
import sys
import base64
from io import BytesIO
from pathlib import Path

from PIL import Image
from fastapi import HTTPException

BACKEND_DIR = Path(__file__).resolve().parents[2]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from models.leafgen_model import generate, pretty, classes


def load_generator_model():
    """Load the trained Conditional DCGAN."""
    from models.leafgen_model import load_generator

    return load_generator()


def generate_leaf_images(model, class_name: str, number_of_images: int):
    """Generate synthetic leaf images for a selected disease."""

    # Validate disease class
    if class_name not in classes:
        raise HTTPException(
            status_code=400,
            detail="Unsupported disease class."
        )

    # Generate images using the original model
    images = generate(
        model,
        class_name,
        n=number_of_images
    )

    results = []

    for index, image in enumerate(images, start=1):

        # Convert NumPy image into PNG bytes
        buffer = BytesIO()

        Image.fromarray(image).save(
            buffer,
            format="PNG"
        )

        # Encode PNG bytes into Base64
        encoded_image = base64.b64encode(
            buffer.getvalue()
        ).decode("utf-8")

        results.append({
            "filename": f"synthetic_leaf_{index}.png",
            "mime_type": "image/png",
            "data_base64": encoded_image,
            "width": 64,
            "height": 64
        })

    return {
        "success": True,
        "class_code": class_name,
        "class_name": pretty(class_name),
        "number_of_images": len(results),
        "images": results
    }
