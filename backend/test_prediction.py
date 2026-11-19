
from pathlib import Path
import sys

import numpy as np
from PIL import Image

# Locate backend directory
BASE_DIR = Path(__file__).resolve().parent

# Locate model helper
MODEL_DIR = BASE_DIR / "models"
sys.path.insert(0, str(MODEL_DIR))

# Import Ameer’s original model functions
from models.leafgen_model import load_classifier, classify, pretty


def main():

    # Locate test image
    image_path = BASE_DIR / "test_images" / "tomato_leaf.jpg"

    # Check whether image exists
    if not image_path.exists():
        print("Error: Test image not found!")
        print("Expected location:", image_path)
        return

    print("Loading leaf image...")

    # Open image and convert to RGB
    image = Image.open(image_path).convert("RGB")

    # Convert image into NumPy uint8 array
    rgb_image = np.asarray(image, dtype=np.uint8)

    print("Image size:", image.size)
    print("Image shape:", rgb_image.shape)

    # Load trained ResNet18 classifier
    print("\nLoading ResNet18...")
    model = load_classifier(variant="real")

    print("Model loaded successfully!")

    # Perform disease classification
    print("\nPredicting plant disease...")

    predictions = classify(
        model,
        rgb_image,
        k=3
    )

    # Display predictions
    print("\n===== LEAFGEN AI RESULTS =====")

    for index, (class_name, probability) in enumerate(
        predictions, start=1
    ):
        print(f"\n{index}. {pretty(class_name)}")
        print(f"   Confidence: {probability * 100:.2f}%")

    print("\nPrediction completed successfully!")


if __name__ == "__main__":
    main()
