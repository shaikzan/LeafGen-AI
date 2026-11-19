# LeafGen AI backend

FastAPI service for plant disease classification and conditional synthetic leaf generation.

## Setup

Use Python 3.10 or newer. From the project root, create and activate a virtual environment, then install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python backend/download_models.py
```

The model downloader fetches the weights and metadata into `backend/models`. The API reads those local files at startup and does not need Hugging Face access after they have been downloaded. Keep `backend/models/leafgen_model.py` in place; it defines the model architecture and inference helpers.

PyTorch publishes platform-specific builds. If the pinned PyTorch packages are unavailable for your OS or Python version, install a compatible `torch` and `torchvision` pair using the [official PyTorch install selector](https://pytorch.org/get-started/locally/) and then install the remaining requirements.

## Run

From the project root:

```powershell
uvicorn app.main:app --app-dir backend --reload
```

Interactive API docs are available at `http://127.0.0.1:8000/docs`.

- `GET /health` checks that the API process is responding.
- `GET /ready` returns ready only after both models load.
- `POST /api/predict` accepts a JPG, JPEG, or PNG image as multipart field `file` (up to 10 MB; decoded dimensions up to 25 megapixels).
- `POST /api/generate` accepts JSON with `class_name` and optional `number_of_images` (1 to 8).

Model loading happens during application startup. Startup fails with a missing asset message if the local model files have not been downloaded.

## Smoke scripts

The `backend/test_*.py` files are manual model smoke scripts. Run them from the project root after setup, for example:

```powershell
python backend/test_prediction.py
python backend/test_generator.py
python backend/test_classifier.py
```

They exercise model loading/inference and are not a pytest suite.
