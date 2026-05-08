from pydantic import BaseModel
from typing import Optional


class InferenceRequest(BaseModel):
    image_id: Optional[str] = None
