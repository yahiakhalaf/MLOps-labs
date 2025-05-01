import pandas as pd
from sklearn.model_selection import train_test_split
import logging
import yaml
from pathlib import Path

logger = logging.getLogger(__name__)

def load_and_split_data(cfg):
    """Load and split data using DVC params.yaml"""
    try:

        logger.info(f"Loading data from {cfg['data']['raw_dir']}/train.csv")
        df = pd.read_csv(f"{cfg['data']['raw_dir']}/train.csv")
        
        # Validate columns
        required_cols = cfg['data']['features']['numerical'] +cfg['data']['features']['categorical'] +  [cfg['data']['target']]
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")
            
        X = df.drop(cfg['data']['target'], axis=1)
        y = df[cfg['data']['target']]
        
        return train_test_split(X, y,test_size=cfg['data']['test_size'],random_state=cfg['model']['params']['random_state'])
        
    except Exception as e:
        logger.error(f"Data loading failed: {str(e)}")
        raise