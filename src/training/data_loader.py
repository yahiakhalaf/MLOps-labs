import pandas as pd
from sklearn.model_selection import train_test_split
import logging
from omegaconf import DictConfig
import hydra
logger = logging.getLogger(__name__)

@hydra.main(config_path="../../config", config_name="config", version_base="1.3")
def load_and_split_data(cfg: DictConfig) :
    """Load and split data with Hydra config"""
    try:
        logger.info(f"Loading data from {cfg.data.train_path}")
        df = pd.read_csv(cfg.data.train_path)
        
        # Ensure columns exist
        required_cols = cfg.preprocessing.numeric_features + cfg.preprocessing.categorical_features + [cfg.data.target]
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")
            
        X = df.drop(cfg.data.target, axis=1)
        y = df[cfg.data.target]
        
        return train_test_split( X, y,test_size=cfg.data.test_size,random_state=cfg.model.random_state)
    except Exception as e:
        logger.error(f"Data loading failed: {str(e)}")
        raise