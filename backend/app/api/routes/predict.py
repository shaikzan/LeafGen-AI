
from fastapi import APIRouter, UploadFile, File, HTTPException, Request
from starlette.concurrency import run_in_threadpool

from app.services.classifier import predict_image

router = APIRouter()


@router.post("/predict")
async def predict(
    request: Request,
    file: UploadFile = File(...)
):
    # Validate file extension
    allowed_extensions = {".jpg", ".jpeg", ".png"}

    extension = (
        "." + file.filename.rsplit(".", 1)[-1].lower()
        if file.filename and "." in file.filename
        else ""
    )

    if extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail="Only JPG, JPEG and PNG images are supported."
        )

    # Read in bounded chunks so oversized requests do not all enter memory.
    max_size = 10 * 1024 * 1024
    image_data = bytearray()
    total_size = 0
    while chunk := await file.read(1024 * 1024):
        total_size += len(chunk)
        if total_size > max_size:
            raise HTTPException(status_code=413, detail="Image size must be below 10 MB.")
        image_data.extend(chunk)
    image_bytes = bytes(image_data)

    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded image is empty.")

    # Get the already-loaded classifier
    model = request.app.state.classifier

    # Perform prediction
    result = await run_in_threadpool(predict_image, model, image_bytes)

    return result
