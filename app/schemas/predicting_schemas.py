from typing import Optional
from pydantic import BaseModel


class PredictionRequest(BaseModel):
    image_path: str
    model_path: str
    model_type: Optional[str] = "simple"  # model factory key
    patch_size: Optional[int] = 9
    n_components: Optional[int] = 30
    num_classes: Optional[int] = 16


class PredictionResponse(BaseModel):
    pred_map_path: str
