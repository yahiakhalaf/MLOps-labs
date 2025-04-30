import joblib
import json
import os
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import hydra
from omegaconf import DictConfig, OmegaConf
from src.training.data_loader import load_and_split_data
from src.training.preprocessor import build_preprocessor
from src.training.evaluate import evaluate_model

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@hydra.main(config_path="../../config", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    try:
        logger.info("Starting training pipeline")
        
        # Load data
        X_train, X_test, y_train, y_test = load_and_split_data(cfg)
        
        # Build model pipeline
        model = Pipeline([
            ("preprocessor", build_preprocessor(cfg)),
            ("classifier", RandomForestClassifier(
                n_estimators=cfg.model.n_estimators,
                max_depth=cfg.model.max_depth,
                random_state=cfg.model.random_state))])
        
        # Train
        logger.info("Training model...")
        model.fit(X_train, y_train)
        
        # Save model
        os.makedirs(os.path.dirname(cfg.paths.model_output), exist_ok=True)
        joblib.dump(model, cfg.paths.model_output)

        metrics = evaluate_model(cfg)
        # Save metrics
        os.makedirs(os.path.dirname(cfg.paths.metrics_output), exist_ok=True)
        with open(cfg.paths.metrics_output, "w") as f:
            json.dump(metrics, f, indent=2)
            
        logger.info(f"Training completed.")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
