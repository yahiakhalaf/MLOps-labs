import joblib
import pandas as pd
import logging
from typing import Union
from config.config import load_config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TitanicPredictor:
    def __init__(self, model_path: str = None):
        """Initialize predictor with trained model."""
        config = load_config()
        self.model_path = model_path or config["paths"]["model_output"]
        
        try:
            self.model = joblib.load(self.model_path)
            logger.info(f"Loaded model from {self.model_path}")
        except Exception as e:
            logger.error(f"Model loading failed: {str(e)}")
            raise

    def predict(self, input_data: Union[dict, pd.DataFrame]) -> int:
        """Make a prediction (0 = died, 1 = survived)."""
        try:
            if isinstance(input_data, dict):
                input_data = self._prepare_input(input_data)
            
            prediction = int(self.model.predict(input_data)[0])
            logger.info(f"Prediction: {prediction}")
            return prediction
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise

    def _prepare_input(self, raw_data: dict) -> pd.DataFrame:
        """Convert raw input to properly formatted DataFrame."""
        config = load_config()
        
        # Ensure all expected features exist
        expected_features = (
            config["preprocessing"]["numeric_features"] +
            config["preprocessing"]["categorical_features"]
        )
        
        # Create DataFrame with all expected columns
        df = pd.DataFrame([raw_data])
        for feat in expected_features:
            if feat not in df.columns:
                df[feat] = None  # Will be imputed by the pipeline
                
        return df

# Example usage
if __name__ == "__main__":
    predictor = TitanicPredictor()
    sample = {
        "Age": 30,
        "Fare": 50,
        "Sex": "female",
        "Embarked": "S"
    }
    print(f"Prediction: {predictor.predict(sample)}")