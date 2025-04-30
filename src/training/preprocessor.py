from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import logging
from omegaconf import DictConfig
import hydra

logger = logging.getLogger(__name__)

@hydra.main(config_path="../../config", config_name="config", version_base="1.3")
def build_preprocessor(cfg: DictConfig) -> ColumnTransformer:
    """Create preprocessor with Hydra config"""
    try:
        logger.info("Building preprocessing pipeline")
        
        # Convert OmegaConf lists to Python lists
        numeric_features = list(cfg.preprocessing.numeric_features)
        categorical_features = list(cfg.preprocessing.categorical_features)

        return ColumnTransformer(
            transformers=[
                ("num", Pipeline([
                    ("imputer", SimpleImputer(strategy=cfg.preprocessing.numeric_strategy)),
                    ("scaler", StandardScaler())
                ]), numeric_features),
                ("cat", Pipeline([
                    ("imputer", SimpleImputer(strategy=cfg.preprocessing.categorical_strategy)),
                    ("onehot", OneHotEncoder(handle_unknown="ignore"))
                ]), categorical_features)
            ],
            remainder="drop"
        )
    except Exception as e:
        logger.error(f"Preprocessor creation failed: {str(e)}")
        raise