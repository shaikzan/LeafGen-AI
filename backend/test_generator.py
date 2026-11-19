
from pathlib import Path
import sys

import numpy as np
from PIL import Image

# Locate backend/models
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

sys.path.insert(0, str(MODEL_DIR))

# Import original generator functions
from models.leafgen_model import load_generator, generate, pretty

def main():

    print("Loading Conditional DCGAN...")

    # Load trained generator
    model = load_generator()

    print("Generator loaded successfully!")

    # Select a supported disease class
    disease = "Apple___Cedar_apple_rust"

    print("\nGenerating synthetic leaf images...")
    print("Selected class:", pretty(disease))

    # Generate 4 synthetic images
    images = generate(
        model,
        disease,
        n=4
    )

    print("Generated image shape:", images.shape)

    # Save generated images
    output_dir = BASE_DIR / "generated_images"
    output_dir.mkdir(exist_ok=True)

    for index, image in enumerate(images, start=1):

        image_path = output_dir / f"leaf_{index}.png"

        Image.fromarray(
            np.asarray(image, dtype=np.uint8)
        ).save(image_path)

        print("Saved:", image_path)

    print("\nSynthetic generation completed!")


if __name__ == "__main__":
    main()
