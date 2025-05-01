from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import logging
import yaml
from pathlib import Path

logger = logging.getLogger(__name__)

def build_preprocessor(cfg):
    """Create preprocessor using DVC params.yaml"""
    try:

        logger.info("Building preprocessing pipeline")
        numeric_features = cfg['data']['features']['numerical']
        categorical_features = cfg['data']['features']['categorical']

        return ColumnTransformer(
            transformers=[
                ("num", Pipeline([
                    ("imputer", SimpleImputer(strategy=cfg['preprocessing']['numeric_strategy'])),
                    ("scaler", StandardScaler())
                ]), numeric_features),
                ("cat", Pipeline([
                    ("imputer", SimpleImputer(strategy=cfg['preprocessing']['categorical_strategy'])),
                    ("onehot", OneHotEncoder(handle_unknown="ignore"))
                ]), categorical_features)
            ],
            remainder="drop"
        )
    except Exception as e:
        logger.error(f"Preprocessor creation failed: {str(e)}")
        raise