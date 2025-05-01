import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import json
import logging
import pandas as pd
import yaml
from pathlib import Path

logger = logging.getLogger(__name__)

def evaluate_model(model,X_test,y_test,cfg) -> dict:
    """Evaluate model using DVC params.yaml"""
    try:

        logger.info("Starting evaluation")

        y_pred = model.predict(X_test)
        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred)),
            "recall": float(recall_score(y_test, y_pred)),
            "f1": float(f1_score(y_test, y_pred))
        }
  
        return metrics

    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        raise