
from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):

    class_name: str = Field(
        ...,
        description="Exact supported disease class code"
    )

    number_of_images: int = Field(
        default=4,
        ge=1,
        le=8,
        description="Number of synthetic images to generate"
    )
