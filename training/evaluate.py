from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import json
import logging
import os

logger = logging.getLogger(__name__)

def evaluate_model(model, X_test, y_test, config: dict) -> dict:
    """Evaluate model and save metrics."""
    logger.info("Evaluating model performance")
    
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred)
    }
    
    logger.info(f"Model metrics: {metrics}")
    
    # Save metrics
    os.makedirs(os.path.dirname(config["paths"]["metrics_output"]), exist_ok=True)
    with open(config["paths"]["metrics_output"], "w") as f:
        json.dump(metrics, f, indent=2)
    
    return metrics