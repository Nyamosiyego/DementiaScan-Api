import sys

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
from pydantic import BaseModel
from typing import Dict
import logging
from app.services.model_service import ModelService

# Configure logging to show in console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Alzheimer's Disease Classification API",
    description="API for classifying brain MRI images using ViT model"
)

# Add CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Define response model
class PredictionResponse(BaseModel):
    predicted_class: str
    confidence_scores: Dict[str, float]

# Initialize model service
model_service = ModelService()

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    model_service.load_model()


from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging

# Configure detailed logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Log request details
        logger.info(f"Received file upload: {file.filename}")
        logger.info(f"Content type: {file.content_type}")

        # Validate file presence
        if not file:
            logger.error("No file received")
            raise HTTPException(status_code=400, detail="No file received")

        # Read file contents
        contents = await file.read()
        logger.info(f"File size: {len(contents)} bytes")

        if len(contents) == 0:
            logger.error("Empty file received")
            raise HTTPException(status_code=400, detail="Empty file received")

        try:
            image = Image.open(io.BytesIO(contents))
            logger.info(f"Successfully opened image: {image.format}, size={image.size}")
        except Exception as e:
            logger.error(f"Failed to open image: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")

        # Process image...
        result = await model_service.predict(image)
        return result

    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}