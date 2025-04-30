import pandas as pd
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger(__name__)

def load_and_split_data(config: dict) -> tuple:
    """Load and split data with logging."""
    try:
        logger.info(f"Loading data from {config['data']['train_path']}")
        df = pd.read_csv(config["data"]["train_path"])
        
        logger.info(f"Splitting data (test_size={config['data']['test_size']})")
        X = df.drop(config["data"]["target"], axis=1)
        y = df[config["data"]["target"]]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config["data"]["test_size"],random_state=config["model"]["random_state"]
)
        
        logger.info(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
        return X_train, X_test, y_train, y_test
        
    except Exception as e:
        logger.error(f"Data loading failed: {str(e)}")
        raise