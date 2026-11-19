
from pathlib import Path
import sys

# Locate the downloaded model directory
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

# Make the helper module importable
sys.path.insert(0, str(MODEL_DIR))

print("Model directory:", MODEL_DIR)
print("Directory exists:", MODEL_DIR.exists())

# Check required files
required_files = [
    "leafgen_model.py",
    "cnn_real.pt",
    "classes.json"
]

for filename in required_files:
    path = MODEL_DIR / filename
    print(filename, ":", path.exists())

# Import the model helper
from models.leafgen_model import load_classifier, classify

print("\nLoading ResNet18 classifier...")

# Load the recommended classifier
classifier = load_classifier()

print("ResNet18 loaded successfully!")
print("Classifier:", type(classifier))
