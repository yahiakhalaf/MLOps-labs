import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import logging
from config.config import load_config
from .data_loader import load_and_split_data
from .preprocessor import build_preprocessor
from .evaluate import evaluate_model

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

def train():
    """Main training pipeline."""
    try:
        config = load_config()
        
        # Data
        X_train, X_test, y_train, y_test = load_and_split_data(config)
        
        # Model
        preprocessor = build_preprocessor(config)
        model = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(n_estimators=config["model"]["n_estimators"],
            max_depth=config["model"]["max_depth"],random_state=config["model"]["random_state"]
            ))
        ])
        
        logger.info("Starting model training...")
        model.fit(X_train, y_train)
        logger.info("Model training completed")
        
        # Save model
        joblib.dump(model, config["paths"]["model_output"])
        logger.info(f"Model saved to {config['paths']['model_output']}")
        
        # Evaluate
        metrics = evaluate_model(model, X_test, y_test, config)
        logger.info(f"Final metrics: {metrics}")
        
        return metrics
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    train()