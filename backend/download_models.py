
from pathlib import Path
from huggingface_hub import hf_hub_download

# Hugging Face repository
REPO_ID = "ameer00712/leafgen-models"

# Always download inside backend/models
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

# Create models directory if it doesn't exist
MODEL_DIR.mkdir(parents=True, exist_ok=True)

FILES = [
    "cnn_real.pt",
    "generator.pt",
    "classes.json",
    "rare.json"
]

for filename in FILES:
    print(f"\nDownloading {filename}...")

    path = hf_hub_download(
        repo_id=REPO_ID,
        filename=filename,
        local_dir=str(MODEL_DIR)
    )

    print(f"Downloaded successfully: {path}")

print("\nAll model files downloaded!")
