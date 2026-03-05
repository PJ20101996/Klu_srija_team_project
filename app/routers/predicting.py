from fastapi import APIRouter, HTTPException
from app.schemas.predicting_schemas import PredictionRequest, PredictionResponse
from app.services.pytorch_training import load_model, predict_image

router = APIRouter()


@router.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """Endpoint to predict on a satellite image sample and return saved map path."""
    try:
        model = load_model(
            request.model_path,
            model_type=request.model_type,
            num_classes=request.num_classes,
        )
        out_path = predict_image(
            model,
            request.image_path,
            patch_size=request.patch_size,
            n_components=request.n_components,
        )
        return PredictionResponse(pred_map_path=out_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
