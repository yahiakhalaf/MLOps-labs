import numpy as np
import litserve as ls
import joblib
import pandas as pd
import logging
from src.deployment.online.requests import InferenceRequest
from src.inference.predictor import TitanicPredictor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class InferenceAPI(ls.LitAPI):
    def setup(self, device="cpu"):
        try:
            self._model = TitanicPredictor()
            self._encoder = {1: "Survived", 0: "Non-Survived"}
            logger.info("Model initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize model: {str(e)}")
            raise

    def decode_request(self, request):
        try:
            # Validate request using Pydantic model
            validated = InferenceRequest(**request)
            # Convert to DataFrame with proper orientation
            df = pd.DataFrame([validated.dict()])
            logger.info("Request decoded successfully")
            return df
        except Exception as e:
            logger.error(f"Request decoding failed: {str(e)}")
            return None

    def predict(self, x):
        try:
            if x is not None and not x.empty:
                predictions = self._model.predict(x)
                logger.info(f"Prediction successful: {predictions}")
                return predictions
            logger.warning("Empty or None input received")
            return None
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            return None

    def encode_response(self, output):
        try:
            if output is None:
                return {
                    "status": "error",
                    "message": "No prediction was generated",
                    "prediction": None
                }
            
            # Ensure output is iterable (convert numpy array if needed)
            if isinstance(output, np.ndarray):
                output = output.tolist()
                
            return {
                "status": "success",
                "message": "Prediction completed",
                "prediction": [self._encoder.get(int(val), "Unknown") for val in output]
            }
        except Exception as e:
            logger.error(f"Response encoding failed: {str(e)}")
            return {
                "status": "error",
                "message": str(e),
                "prediction": None
            }