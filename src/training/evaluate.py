import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import json
import logging
import os
import hydra
from omegaconf import DictConfig
from typing import Any
import pandas as pd

logger = logging.getLogger(__name__)

@hydra.main(config_path="../../config", config_name="config", version_base="1.3")
def evaluate_model(config: DictConfig) -> dict:
    """Evaluate model performance and save metrics."""
    try:
        logger.info("Starting model evaluation")
        
        # Load test data
        logger.info(f"Loading test data from {config.data.train_path}")
        df = pd.read_csv(config.data.train_path)
        X_test = df.drop(config.data.target, axis=1)
        y_test = df[config.data.target]
        
        # Load trained model
        logger.info(f"Loading model from {config.paths.model_output}")
        model = joblib.load(config.paths.model_output)
        
        # Make predictions
        logger.info("Making predictions on test data")
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred)),
            "recall": float(recall_score(y_test, y_pred)),
            "f1": float(f1_score(y_test, y_pred))
        }

        # Log and save metrics
        logger.info(f"Model metrics: {metrics}")
        os.makedirs(os.path.dirname(config.paths.metrics_output), exist_ok=True)
        with open(config.paths.metrics_output, "w") as f:
            json.dump(metrics, f, indent=2)
            
        logger.info(f"Metrics saved to {config.paths.metrics_output}")
        return metrics

    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    # Initialize logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("logs/evaluation.log"),
            logging.StreamHandler()
        ]
    )
    
    evaluate_model()