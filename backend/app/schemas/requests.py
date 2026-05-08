from pydantic import BaseModel
from typing import Optional


class InferenceRequest(BaseModel):
    """Request schema for inference endpoint."""
    image_id: Optional[str] = None