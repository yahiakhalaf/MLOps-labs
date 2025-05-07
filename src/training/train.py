import joblib
import json
import os
import logging
import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from training.data_loader import load_and_split_data
from training.preprocessor import build_preprocessor
from training.evaluate import evaluate_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from utils.mlflow_setup import setup_mlflow
import mlflow
import mlflow.sklearn

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

def main():
    try:
        logger.info("Starting training pipeline")

        # Load DVC config
        with open("params.yaml") as f:
            cfg = yaml.safe_load(f)

        setup_mlflow()

        # Load data
        X_train, X_test, y_train, y_test = load_and_split_data(cfg)

        # Build model pipeline
        model = Pipeline([
            ("preprocessor", build_preprocessor(cfg)),
            ("classifier", RandomForestClassifier(
                n_estimators=cfg['model']['params']['n_estimators'],
                max_depth=cfg['model']['params']['max_depth'],
                random_state=cfg['model']['params']['random_state']))
        ])

        # Save training data
        X_train.to_csv("data/processed/X_train.csv", index=False)
        y_train.to_csv("data/processed/y_train.csv", index=False)

        with mlflow.start_run():
            logger.info("Training model...")
            model.fit(X_train, y_train)

            # Save model locally
            Path(cfg['paths']['model_dir']).mkdir(exist_ok=True)
            joblib.dump(model, cfg['paths']['model_output'])

            # Evaluate
            metrics = evaluate_model(model, X_test, y_test, cfg)
            with open(cfg['paths']['metrics_file'], 'w') as f:
                json.dump(metrics, f, indent=2)

            # Log metrics and model to MLflow
            mlflow.log_params(cfg['model']['params'])
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, "model")

            print("\033[92m", metrics, "\033[0m")
            logger.info("Training completed")

    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
