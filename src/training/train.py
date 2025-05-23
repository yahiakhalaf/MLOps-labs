import joblib
import json
import os
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
import yaml
from pathlib import Path
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

def main():
    try:
        logger.info("Starting training pipeline")
        
        # Load DVC config
        with open("params.yaml") as f:
            cfg = yaml.safe_load(f)
        
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
        X_train.to_csv("data/processed/X_train.csv", index=False)
        y_train.to_csv("data/processed/y_train.csv", index=False)
        # Train and save
        logger.info("Training model...")
        model.fit(X_train, y_train)
        
        Path(cfg['paths']['model_dir']).mkdir(exist_ok=True)
        joblib.dump(model, cfg['paths']['model_output'])

        # Evaluate
        metrics = evaluate_model(model,X_test,y_test,cfg)
        with open(cfg['paths']['metrics_file'], 'w') as f:
            json.dump(metrics, f, indent=2)
        print("\033[92m", metrics, "\033[0m")
        logger.info("Training completed")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()