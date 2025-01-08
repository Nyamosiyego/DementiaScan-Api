from PIL import Image
import torch
from transformers import ViTForImageClassification
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from fastapi import HTTPException
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelService:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.id2label = {
            0: "Mild Dementia",
            1: "Moderate Dementia",
            2: "Non Demented",
            3: "Very mild Dementia"
        }
        self.transform = Compose([
            Resize((224, 224)),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def load_model(self):
        if self.model is None:
            try:
                logger.info("Loading model...")
                self.model = ViTForImageClassification.from_pretrained(
                    'fawadkhan/ViT_FineTuned_on_ImagesOASIS'
                )
                self.model.to(self.device)
                self.model.eval()
                logger.info("Model loaded successfully")
            except Exception as e:
                logger.error(f"Error loading model: {str(e)}")
                raise RuntimeError("Failed to load model")

    async def predict(self, image: Image.Image) -> dict:
        try:
            # Preprocess image
            input_tensor = self.transform(image).unsqueeze(0)
            input_tensor = input_tensor.to(self.device)

            # Get predictions
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()

            # Get confidence scores for all classes
            confidence_scores = {
                self.id2label[i]: float(probabilities[0][i])
                for i in range(len(self.id2label))
            }

            return {
                "predicted_class": self.id2label[predicted_class],
                "confidence_scores": confidence_scores
            }
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise HTTPException(status_code=500, detail="Prediction failed")