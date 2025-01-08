# app/schemas/request_models.py
from pydantic import BaseModel, Field, ConfigDict
from typing import Dict, List, Optional
from datetime import datetime

class BaseResponseModel(BaseModel):
    """Base model with configuration to avoid namespace conflicts"""
    model_config = ConfigDict(protected_namespaces=())

class HealthResponse(BaseResponseModel):
    """Health check response model"""
    status: str
    is_model_loaded: bool = Field(..., description="Indicates if the model is loaded")  # Renamed from model_loaded
    timestamp: str
    uptime: float
    environment: str = "production"

class PredictionRequest(BaseResponseModel):
    """Prediction request model"""
    image: str = Field(..., description="Base64 encoded image string")
    include_probabilities: bool = Field(True, description="Include probabilities for all classes")

class PredictionResponse(BaseResponseModel):
    """Prediction response model"""
    predicted_class: str
    confidence: float
    class_probabilities: Optional[Dict[str, float]]
    prediction_time: float
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

class ModelMetadata(BaseResponseModel):
    """Model information response"""
    name: str
    version: str
    last_updated: str
    input_shape: tuple
    classes: List[str]
    architecture_type: str = Field(..., description="Model architecture type")  # Renamed from model_type
    total_predictions: int = 0

class ErrorResponse(BaseResponseModel):
    """Error response model"""
    detail: str
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    error_code: Optional[str] = None