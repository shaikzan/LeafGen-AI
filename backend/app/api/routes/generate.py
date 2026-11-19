
from fastapi import APIRouter, Request

from app.schemas.generator import GenerateRequest
from app.services.generator import generate_leaf_images

router = APIRouter()


@router.post("/generate")
def generate(
    data: GenerateRequest,
    request: Request
):
    # Access the loaded DCGAN model
    model = request.app.state.generator

    # Generate synthetic images
    result = generate_leaf_images(
        model=model,
        class_name=data.class_name,
        number_of_images=data.number_of_images
    )

    return result
